import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import time
import math
import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from collections import deque
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Controller as MouseController

cv2.setUseOptimized(True)
torch.set_num_threads(2)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

import train_mlp

CONFIG_FILE = "config.json"
MODEL_FILE = os.path.join("models", "gesture_mlp.pth")


# ==============================================================================
# 0. HIGH-SPEED THREADED WEBCAM STREAM (WITH HARDWARE CAMERA ON/OFF CONTROL)
# ==============================================================================
class ThreadedCameraStream:
    """
    Non-blocking background camera reader with full hardware power-off/on support.
    When stop_camera() is called, self.cap.release() physically disconnects the
    webcam sensor (turning off the laptop camera LED).
    """
    def __init__(self, src=0, width=640, height=480, start_enabled=True):
        self.src = src
        self.width = width
        self.height = height
        self.cap = None
        self.is_cam_enabled = bool(start_enabled)

        self.ret = False
        self.frame = None
        self.frame_id = 0
        self.measured_cam_fps = 0.0
        self._last_cam_time = time.perf_counter()

        self.lock = threading.Lock()
        self.running = True

        if self.is_cam_enabled:
            self._open_hardware_camera()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _open_hardware_camera(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass

        # Use default Windows backend first (often 30 FPS where DSHOW locks to 10 FPS),
        # with fallback to CAP_DSHOW if needed.
        cap = cv2.VideoCapture(self.src)
        if not cap.isOpened() and os.name == "nt":
            cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

        mjpg = cv2.VideoWriter_fourcc(*"MJPG")
        cap.set(cv2.CAP_PROP_FOURCC, mjpg)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        ret, first_frame = cap.read()
        with self.lock:
            self.cap = cap
            self.ret = bool(ret and first_frame is not None)
            self.frame = cv2.flip(first_frame, 1) if self.ret else None
            self.measured_cam_fps = 30.0 if self.ret else 0.0
            self._last_cam_time = time.perf_counter()

    def set_camera_enabled(self, enabled):
        enabled = bool(enabled)
        if enabled == self.is_cam_enabled:
            return
        self.is_cam_enabled = enabled
        if not enabled:
            with self.lock:
                self.ret = False
                self.frame = None
                self.measured_cam_fps = 0.0
                if self.cap is not None:
                    try:
                        self.cap.release()
                    except Exception:
                        pass
                    self.cap = None
        else:
            threading.Thread(target=self._open_hardware_camera, daemon=True).start()

    def _reader_loop(self):
        while self.running:
            if not self.is_cam_enabled or self.cap is None or not self.cap.isOpened():
                time.sleep(0.02)
                continue
            try:
                ret, raw = self.cap.read()
            except Exception:
                ret, raw = False, None

            if ret and raw is not None and self.is_cam_enabled:
                now = time.perf_counter()
                dt = max(now - self._last_cam_time, 1e-4)
                self._last_cam_time = now
                inst_fps = min(1.0 / dt, 60.0)
                flipped = cv2.flip(raw, 1)
                with self.lock:
                    self.ret = True
                    self.frame = flipped
                    self.frame_id += 1
                    self.measured_cam_fps = 0.82 * self.measured_cam_fps + 0.18 * inst_fps
            else:
                time.sleep(0.002)

    def read(self):
        with self.lock:
            if not self.is_cam_enabled or self.frame is None:
                return False, None, self.frame_id, 0.0
            return self.ret, self.frame, self.frame_id, self.measured_cam_fps

    def release(self):
        self.running = False
        self.is_cam_enabled = False
        with self.lock:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass
                self.cap = None


# ==============================================================================
# 1. HOVER TOOLTIP WIDGET FOR 'ⓘ' INFO BADGES
# ==============================================================================
class HoverTooltip:
    """
    Displays a sleek dark floating tooltip card when the user hovers over
    any 'ⓘ' info icon in the Settings view.
    """
    def __init__(self, widget, title, body_text):
        self.widget = widget
        self.title = title
        self.body_text = body_text
        self.tip_win = None
        self.widget.bind("<Enter>", self._show)
        self.widget.bind("<Leave>", self._hide)

    def _show(self, event=None):
        if self.tip_win is not None:
            return
        x = self.widget.winfo_rootx() + 26
        y = self.widget.winfo_rooty() - 6

        self.tip_win = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        tw.attributes("-topmost", True)

        outer = tk.Frame(tw, bg="#0F172A", highlightthickness=1, highlightbackground="#334155", padx=12, pady=10)
        outer.pack()

        tk.Label(
            outer, text=self.title, font=("Segoe UI", 9, "bold"),
            fg="#60A5FA", bg="#0F172A", anchor="w", justify="left"
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            outer, text=self.body_text, font=("Segoe UI", 8),
            fg="#F8FAFC", bg="#0F172A", anchor="w", justify="left", wraplength=380
        ).pack(anchor="w")

    def _hide(self, event=None):
        if self.tip_win is not None:
            try:
                self.tip_win.destroy()
            except Exception:
                pass
            self.tip_win = None


# ==============================================================================
# 2. MODULAR INTEGRATION INTERFACE (SHARED CONFIG & IPC STATE MANAGER)
# ==============================================================================
class IPCConfigManager:
    DEFAULT_CONFIG = {
        "camera_enabled": True,
        "recognition_enabled": True,
        "min_detection_confidence": 0.65,
        "min_tracking_confidence": 0.65,
        "frame_skip": 2,
        "model_complexity": 0,          # 0 = Lite (Fastest ms on laptops), 1 = Full
        "low_light_clahe": False,       # Real OpenCV YCrCb CLAHE contrast boost
        "show_skeleton_overlay": True,
        "swipe_cooldown": 0.42,
        "zoom_sensitivity": 0.050,
        "enable_zoom_optional": True,
        "active_tab": "live",           # 'live' or 'settings'
        "gesture_mappings": {
            "Right Swipe": "Right Arrow",
            "Left Swipe": "Left Arrow",
            "Open Palm": "Key B",
            "Pinch In / Out": "Ctrl + / -",
            "Index Finger Point": "Key L"
        },
        "gesture_states": {
            "Right Swipe": True,
            "Left Swipe": True,
            "Open Palm": False,
            "Pinch In / Out": True,
            "Index Finger Point": True
        }
    }

    def __init__(self, filepath=CONFIG_FILE):
        self.filepath = filepath
        self.state = self.load()
        self.last_sync_time = time.time()

    def load(self):
        if not os.path.exists(self.filepath):
            self.save(self.DEFAULT_CONFIG.copy())
            return self.DEFAULT_CONFIG.copy()
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            merged = self.DEFAULT_CONFIG.copy()
            merged.update(data)
            if "gesture_mappings" not in data:
                merged["gesture_mappings"] = self.DEFAULT_CONFIG["gesture_mappings"].copy()
            else:
                for k, v in self.DEFAULT_CONFIG["gesture_mappings"].items():
                    if k not in merged["gesture_mappings"]:
                        merged["gesture_mappings"][k] = v
            if "gesture_states" not in data:
                merged["gesture_states"] = self.DEFAULT_CONFIG["gesture_states"].copy()
            if merged.get("active_tab") == "mapping":
                merged["active_tab"] = "settings"
            return merged
        except Exception:
            return self.DEFAULT_CONFIG.copy()

    def save(self, cfg=None):
        if cfg is not None:
            self.state = cfg
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            print(f"[WARN] IPC Config Save Error: {e}")

    def poll_if_needed(self, interval=2.0):
        if (time.time() - self.last_sync_time) > interval:
            self.state = self.load()
            self.last_sync_time = time.time()
        return self.state


# ==============================================================================
# 3. OS HARDWARE ABSTRACTION LAYER (KEYBOARD & ZOOM EMULATION)
# ==============================================================================
class OSActionDispatcher:
    def __init__(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()

    def execute_zoom(self, ticks):
        if ticks == 0:
            return "None"

        steps = min(abs(int(ticks)), 3)

        if HAS_PYAUTOGUI:
            pyautogui.keyDown('ctrl')
            try:
                scroll_units = 120 * steps if ticks > 0 else -120 * steps
                pyautogui.scroll(scroll_units)
                key_sym = '+' if ticks > 0 else '-'
                for _ in range(steps):
                    pyautogui.press(key_sym)
            finally:
                pyautogui.keyUp('ctrl')
        else:
            with self.keyboard.pressed(Key.ctrl):
                self.mouse.scroll(0, steps if ticks > 0 else -steps)
                key_char = '=' if ticks > 0 else '-'
                for _ in range(steps):
                    self.keyboard.press(key_char)
                    self.keyboard.release(key_char)

        return "Ctrl + Plus (Zoom In)" if ticks > 0 else "Ctrl + Minus (Zoom Out)"

    def execute_shortcut(self, gesture_name, mappings, states=None):
        if states and not states.get(gesture_name, True):
            return "Disabled"

        mapped_key = mappings.get(gesture_name, "")
        if not mapped_key:
            return "None"

        low = mapped_key.lower()
        if "right" in low:
            if HAS_PYAUTOGUI:
                pyautogui.press('right')
            else:
                self.keyboard.press(Key.right)
                self.keyboard.release(Key.right)
            return "Right Arrow (Next Slide)"
        elif "left" in low:
            if HAS_PYAUTOGUI:
                pyautogui.press('left')
            else:
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
            return "Left Arrow (Previous Slide)"
        elif "ctrl" in low:
            return self.execute_zoom(1)
        elif "key " in low:
            char = low.replace("key ", "").strip()
            if len(char) == 1:
                if HAS_PYAUTOGUI:
                    pyautogui.press(char)
                else:
                    self.keyboard.press(char)
                    self.keyboard.release(char)
                return f"Key '{char.upper()}'"
        elif len(mapped_key) == 1:
            self.keyboard.press(mapped_key)
            self.keyboard.release(mapped_key)
            return f"Key '{mapped_key.upper()}'"

        return mapped_key


# ==============================================================================
# 4. SIGNAL PROCESSING: EXPONENTIAL MOVING AVERAGE (EMA)
# ==============================================================================
class ExponentialMovingAverage:
    def __init__(self, alpha=0.42, fast_alpha=0.80, velocity_threshold=0.18):
        self.alpha = float(alpha)
        self.fast_alpha = float(fast_alpha)
        self.velocity_threshold = float(velocity_threshold)
        self.value = None
        self.prev_time = None

    def reset(self):
        self.value = None
        self.prev_time = None

    def filter(self, measurement, timestamp=None):
        if timestamp is None:
            timestamp = time.perf_counter()
        if self.value is None:
            self.value = float(measurement)
            self.prev_time = timestamp
            return self.value

        dt = max(timestamp - self.prev_time, 1e-4)
        velocity = abs(measurement - self.value) / dt

        blend = min(velocity / self.velocity_threshold, 1.0)
        effective_alpha = self.alpha + blend * (self.fast_alpha - self.alpha)

        self.value = effective_alpha * float(measurement) + (1.0 - effective_alpha) * self.value
        self.prev_time = timestamp
        return self.value


# ==============================================================================
# 5. TABLET-STYLE BIDIRECTIONAL PINCH-IN & PINCH-OUT CONTROLLER
# ==============================================================================
class TabletPinchZoomController:
    STATE_IDLE = "IDLE"
    STATE_ARMED = "ARMED"
    STATE_ZOOM_IN = "PINCH OUT (ZOOM IN)"
    STATE_ZOOM_OUT = "PINCH IN (ZOOM OUT)"

    def __init__(self):
        self.ratio_ema = ExponentialMovingAverage(alpha=0.45, fast_alpha=0.82)
        self.state = self.STATE_IDLE
        self.anchor_ratio = 0.0
        self.prev_smoothed_ratio = None
        self.zoom_accumulator = 0.0

        self.RELEASE_RATIO = 1.38
        self.STEP_SENSITIVITY = 0.050
        self.DEADZONE_DELTA = 0.024
        self.DEADZONE_FRAME = 0.005

        self.arm_start_time = 0.0
        self.ARM_HOLD_TIME = 0.06
        self.last_zoom_dir = 0
        self.last_active_time = 0.0

    def reset(self):
        self.state = self.STATE_IDLE
        self.ratio_ema.reset()
        self.prev_smoothed_ratio = None
        self.zoom_accumulator = 0.0
        self.arm_start_time = 0.0
        self.last_zoom_dir = 0

    @staticmethod
    def _dist(lm1, lm2):
        return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    def _is_pinch_zoom_posture(self, landmarks, smoothed_ratio):
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        thumb_mcp = landmarks.landmark[2]
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]

        dx_index = index_tip.x - wrist.x
        dy_index = index_tip.y - wrist.y
        if abs(dx_index) > 0.11 and abs(dx_index) > abs(dy_index) * 1.15:
            return False

        index_reach = self._dist(index_tip, wrist) / max(self._dist(index_mcp, wrist), 1e-4)
        thumb_reach = self._dist(thumb_tip, wrist) / max(self._dist(thumb_mcp, wrist), 1e-4)
        if index_reach < 0.72 and thumb_reach < 0.85:
            return False

        ring_curled = self._dist(ring_tip, wrist) < self._dist(ring_pip, wrist) * 1.20
        pinky_curled = self._dist(pinky_tip, wrist) < self._dist(pinky_pip, wrist) * 1.20
        middle_curled = self._dist(middle_tip, wrist) < self._dist(middle_pip, wrist) * 1.22

        if self.state != self.STATE_IDLE:
            all_open = (not ring_curled) and (not pinky_curled) and (not middle_curled) and (smoothed_ratio > 0.70)
            if all_open or smoothed_ratio > self.RELEASE_RATIO:
                return False
            return True

        if smoothed_ratio <= 0.44:
            return True
        if (ring_curled and pinky_curled and middle_curled) and smoothed_ratio <= 1.05:
            return True

        return False

    def update(self, landmarks, norm_velocity_x, mp_hand_score=0.95, step_sensitivity=0.050):
        self.STEP_SENSITIVITY = float(step_sensitivity)
        now = time.perf_counter()
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        middle_mcp = landmarks.landmark[9]
        index_mcp = landmarks.landmark[5]
        pinky_mcp = landmarks.landmark[17]

        palm_bone = self._dist(wrist, middle_mcp)
        palm_knuckle_span = self._dist(index_mcp, pinky_mcp)
        hand_scale = max(0.65 * palm_bone + 0.35 * palm_knuckle_span, 1e-4)

        raw_pinch = self._dist(thumb_tip, index_tip)
        normalized_ratio = raw_pinch / hand_scale
        smoothed_ratio = self.ratio_ema.filter(normalized_ratio, now)

        if abs(norm_velocity_x) > 1.80:
            self.reset()
            return self.STATE_IDLE, 0, smoothed_ratio, "", 0.0

        if not self._is_pinch_zoom_posture(landmarks, smoothed_ratio):
            self.reset()
            return self.STATE_IDLE, 0, smoothed_ratio, "", 0.0

        zoom_ticks = 0
        status_label = ""
        dynamic_conf = 0.0

        if self.state == self.STATE_IDLE:
            if self.arm_start_time == 0.0:
                self.arm_start_time = now
            elif (now - self.arm_start_time) >= self.ARM_HOLD_TIME:
                self.state = self.STATE_ARMED
                self.anchor_ratio = max(smoothed_ratio, 0.22)
                self.prev_smoothed_ratio = smoothed_ratio
                self.zoom_accumulator = 0.0
                status_label = "PINCH LOCKED (SPREAD / PINCH)"
                dynamic_conf = float(np.clip(mp_hand_score * 96.0, 82.0, 98.5))

        elif self.state == self.STATE_ARMED:
            status_label = "PINCH LOCKED (SPREAD / PINCH)"
            delta_from_anchor = smoothed_ratio - self.anchor_ratio
            self.prev_smoothed_ratio = smoothed_ratio
            dynamic_conf = float(np.clip(mp_hand_score * 96.0 + abs(delta_from_anchor) * 20.0, 84.0, 99.1))

            if abs(delta_from_anchor) >= self.DEADZONE_DELTA:
                self.zoom_accumulator = delta_from_anchor
                if delta_from_anchor > 0:
                    self.state = self.STATE_ZOOM_IN
                else:
                    self.state = self.STATE_ZOOM_OUT

        elif self.state in (self.STATE_ZOOM_IN, self.STATE_ZOOM_OUT):
            delta = smoothed_ratio - self.prev_smoothed_ratio
            self.prev_smoothed_ratio = smoothed_ratio

            if abs(delta) < self.DEADZONE_FRAME:
                delta = 0.0

            self.zoom_accumulator += delta

            if abs(self.zoom_accumulator) >= self.STEP_SENSITIVITY:
                zoom_ticks = int(self.zoom_accumulator / self.STEP_SENSITIVITY)
                self.zoom_accumulator -= zoom_ticks * self.STEP_SENSITIVITY
                self.last_zoom_dir = 1 if zoom_ticks > 0 else -1
                self.last_active_time = now

            displacement_strength = min(abs(smoothed_ratio - self.anchor_ratio) / 0.35, 1.0)
            dynamic_conf = float(np.clip((0.75 * mp_hand_score + 0.25 * displacement_strength) * 100.0, 85.0, 99.8))

            scale_pct = int(np.clip((smoothed_ratio / self.anchor_ratio) * 100, 40, 300))
            if self.last_zoom_dir >= 0:
                self.state = self.STATE_ZOOM_IN
                status_label = f"PINCH OUT -> ZOOM IN (+{scale_pct}%)"
            else:
                self.state = self.STATE_ZOOM_OUT
                status_label = f"PINCH IN -> ZOOM OUT ({scale_pct}%)"

        return self.state, zoom_ticks, smoothed_ratio, status_label, dynamic_conf


# ==============================================================================
# 6. HYBRID STATIC-DIRECTION & SCALE-INVARIANT VELOCITY GESTURE ENGINE
# ==============================================================================
class VisionGestureEngine:
    def __init__(self, model_path=MODEL_FILE):
        if not os.path.exists(model_path):
            train_mlp.main()

        checkpoint = torch.load(model_path)
        self.classes = checkpoint["classes"]
        self.model = train_mlp.WAVE_MLP(input_dim=42, num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.trajectory = deque(maxlen=5)
        self.prediction_buffer = deque(maxlen=2)
        self.last_trigger_time = 0.0
        self.last_swipe_dir = None
        self.last_swipe_time = 0.0
        self.RECOIL_LOCK_SEC = 0.85

    def reset_tracking(self):
        self.trajectory.clear()
        self.prediction_buffer.clear()

    @staticmethod
    def is_within_safe_frame_margin(landmarks, margin=0.035):
        wrist = landmarks.landmark[0]
        return (margin <= wrist.x <= (1.0 - margin)) and (margin <= wrist.y <= (1.0 - margin))

    def is_recoil_blocked(self, candidate_gesture, is_deliberate_point, now_time):
        if candidate_gesture not in ("Left Swipe", "Right Swipe"):
            return False
        if self.last_swipe_dir is None:
            return False
        if candidate_gesture != self.last_swipe_dir and not is_deliberate_point:
            if (now_time - self.last_swipe_time) < self.RECOIL_LOCK_SEC:
                return True
        return False

    @staticmethod
    def is_strict_closed_fist(landmarks, zoom_active=False):
        if zoom_active:
            return False

        wrist = landmarks.landmark[0]
        thumb_tip, thumb_mcp = landmarks.landmark[4], landmarks.landmark[2]
        index_tip, index_mcp = landmarks.landmark[8], landmarks.landmark[5]
        middle_tip, middle_mcp = landmarks.landmark[12], landmarks.landmark[9]
        ring_tip, ring_mcp = landmarks.landmark[16], landmarks.landmark[13]
        pinky_tip, pinky_mcp = landmarks.landmark[20], landmarks.landmark[17]

        palm_len = max(math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y), 1e-4)

        pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) / palm_len
        index_ext = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) / palm_len
        if pinch_dist < 0.40 and index_ext > 0.75:
            return False

        fingers_folded = (
            math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) < math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y) * 0.88 and
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) < math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y) * 0.88 and
            math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < math.hypot(ring_mcp.x - wrist.x, ring_mcp.y - wrist.y) * 0.88 and
            math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < math.hypot(pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y) * 0.88
        )
        thumb_folded = math.hypot(thumb_tip.x - wrist.x, thumb_tip.y - wrist.y) < math.hypot(thumb_mcp.x - wrist.x, thumb_mcp.y - wrist.y) * 1.25
        return fingers_folded and thumb_folded

    @staticmethod
    def detect_static_direction_pose(landmarks, mp_hand_score=0.95):
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]

        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y
        index_len = math.hypot(dx, dy)
        palm_len = max(math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y), 1e-4)

        extension_ratio = index_len / palm_len
        horizontal_dominance = abs(dx) / max(abs(dy), 1e-4)
        norm_dx = dx / palm_len

        if extension_ratio > 1.12 and horizontal_dominance > 1.08:
            angle_clarity = min((horizontal_dominance - 1.0) / 1.5, 1.0)
            reach_clarity = min((extension_ratio - 1.0) / 0.8, 1.0)
            dyn_conf = float(np.clip((0.60 * mp_hand_score + 0.25 * angle_clarity + 0.15 * reach_clarity), 0.82, 0.996))

            if norm_dx < -0.85:
                return "Left Swipe", "DIRECTION: LEFT", dyn_conf
            elif norm_dx > 0.85:
                return "Right Swipe", "DIRECTION: RIGHT", dyn_conf

        if norm_dx < -0.60:
            return None, "DIRECTION: LEFT", float(mp_hand_score * 0.75)
        elif norm_dx > 0.60:
            return None, "DIRECTION: RIGHT", float(mp_hand_score * 0.75)
        return None, "DIRECTION: CENTER", float(mp_hand_score * 0.85)

    def compute_velocity_and_swipe(self, landmarks, now_time, mp_hand_score=0.95):
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        palm_len = max(math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y), 0.04)

        self.trajectory.append((wrist.x, now_time, palm_len))
        if len(self.trajectory) < 3:
            return 0.0, None, 0.0

        x0, t0, _ = self.trajectory[0]
        x1, t1, p1 = self.trajectory[-1]
        dt = max(t1 - t0, 1e-3)
        raw_dx = x1 - x0

        norm_dx = raw_dx / p1
        norm_velocity_x = norm_dx / dt

        diffs = [self.trajectory[i][0] - self.trajectory[i - 1][0] for i in range(1, len(self.trajectory))]
        min_step = 0.02 * p1
        consistent_right = sum(1 for d in diffs if d > min_step) >= len(diffs) - 1
        consistent_left = sum(1 for d in diffs if d < -min_step) >= len(diffs) - 1

        speed_factor = min(abs(norm_velocity_x) / 5.5, 1.0)
        dist_factor = min(abs(norm_dx) / 1.1, 1.0)
        dyn_conf = float(np.clip(0.55 * mp_hand_score + 0.30 * speed_factor + 0.15 * dist_factor, 0.84, 0.998))

        if norm_dx > 0.42 and norm_velocity_x > 2.10 and consistent_right:
            return norm_velocity_x, "Right Swipe", dyn_conf
        elif norm_dx < -0.42 and norm_velocity_x < -2.10 and consistent_left:
            return norm_velocity_x, "Left Swipe", dyn_conf

        return norm_velocity_x, None, 0.0

    def classify_static_mlp(self, landmarks, gesture_states=None):
        features_42 = train_mlp.extract_42_features(landmarks).reshape(1, -1)
        tensor_input = torch.from_numpy(features_42)
        with torch.no_grad():
            outputs = self.model(tensor_input)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.classes[pred_idx]
        conf = float(probs[pred_idx])

        if label in ("Left Swipe", "Right Swipe") or "pinch" in label.lower():
            return "Neutral", conf
        if label == "Open Palm" and gesture_states and not gesture_states.get("Open Palm", False):
            return "Neutral", conf

        return label, conf


# ==============================================================================
# 7. HIGH-DPI PIL ICON & GRAPHICS FACTORY
# ==============================================================================
class GraphicsFactory:
    @staticmethod
    def create_wave_logo(size=34):
        scale = 4
        S = size * scale
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)

        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.24), fill="#136DEC")

        cx, cy = int(S * 0.50), int(S * 0.60)
        palm_r = int(S * 0.16)
        d.ellipse([cx - palm_r, cy - palm_r, cx + palm_r, cy + palm_r], fill="#FFFFFF")
        d.rounded_rectangle([cx - palm_r, cy - int(S * 0.08), cx + palm_r, cy + int(S * 0.14)], radius=10, fill="#FFFFFF")

        fw = int(S * 0.065)
        fingers = [
            (cx - int(S * 0.14), cy - int(S * 0.28), cx - int(S * 0.14), cy),
            (cx - int(S * 0.045), cy - int(S * 0.34), cx - int(S * 0.045), cy),
            (cx + int(S * 0.05), cy - int(S * 0.31), cx + int(S * 0.05), cy),
            (cx + int(S * 0.14), cy - int(S * 0.23), cx + int(S * 0.14), cy),
            (cx - int(S * 0.24), cy - int(S * 0.08), cx - int(S * 0.10), cy + int(S * 0.04))
        ]
        for x0, y0, x1, y1 in fingers:
            d.line([x0, y0, x1, y1], fill="#FFFFFF", width=fw, joint="curve")
            r = fw // 2
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill="#FFFFFF")

        arc_w = max(4, int(S * 0.045))
        d.arc([int(S * 0.12), int(S * 0.14), int(S * 0.45), int(S * 0.50)], start=195, end=260, fill="#FFFFFF", width=arc_w)
        d.arc([int(S * 0.55), int(S * 0.14), int(S * 0.88), int(S * 0.50)], start=280, end=345, fill="#FFFFFF", width=arc_w)

        return ImageTk.PhotoImage(img.resize((size, size), Image.Resampling.LANCZOS))

    @staticmethod
    def create_avatar_icon(size=32):
        scale = 4
        S = size * scale
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.ellipse([0, 0, S - 1, S - 1], fill="#136DEC")
        hr = int(S * 0.14)
        cx, cy = S // 2, int(S * 0.38)
        d.ellipse([cx - hr, cy - hr, cx + hr, cy + hr], outline="#FFFFFF", width=int(S * 0.06))
        sr = int(S * 0.25)
        d.arc([cx - sr, int(S * 0.54), cx + sr, int(S * 0.96)], start=190, end=350, fill="#FFFFFF", width=int(S * 0.06))
        return ImageTk.PhotoImage(img.resize((size, size), Image.Resampling.LANCZOS))

    @staticmethod
    def create_toggle_switch(w=42, h=24, is_on=True):
        scale = 4
        W, H = w * scale, h * scale
        img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        bg = "#136DEC" if is_on else "#CBD5E1"
        d.rounded_rectangle([0, 0, W - 1, H - 1], radius=H // 2, fill=bg)
        pad = int(H * 0.12)
        knob_d = H - 2 * pad
        kx = (W - pad - knob_d) if is_on else pad
        d.ellipse([kx, pad, kx + knob_d, pad + knob_d], fill="#FFFFFF")
        return ImageTk.PhotoImage(img.resize((w, h), Image.Resampling.LANCZOS))

    @staticmethod
    def create_gesture_icon(kind, size=40):
        scale = 4
        S = size * scale
        img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, S - 1, S - 1], radius=int(S * 0.22), fill="#E8F1FD")

        fg = "#136DEC"
        lw = int(S * 0.055)
        cx, cy = S // 2, S // 2

        if kind in ("Right Swipe", "Left Swipe"):
            d.arc([int(S * 0.24), int(S * 0.20), int(S * 0.76), int(S * 0.55)], start=200, end=340, fill=fg, width=lw)
            if kind == "Right Swipe":
                ax, ay = int(S * 0.73), int(S * 0.32)
                d.line([ax - 14, ay - 10, ax, ay], fill=fg, width=lw)
                d.line([ax - 14, ay + 10, ax, ay], fill=fg, width=lw)
            else:
                ax, ay = int(S * 0.27), int(S * 0.32)
                d.line([ax + 14, ay - 10, ax, ay], fill=fg, width=lw)
                d.line([ax + 14, ay + 10, ax, ay], fill=fg, width=lw)
            d.rounded_rectangle([cx - int(S * 0.13), cy + int(S * 0.02), cx + int(S * 0.13), cy + int(S * 0.26)],
                                radius=12, outline=fg, width=lw)
            d.line([cx, cy + int(S * 0.02), cx, cy - int(S * 0.14)], fill=fg, width=lw)

        elif kind == "Open Palm":
            d.rounded_rectangle([cx - int(S * 0.15), cy - int(S * 0.02), cx + int(S * 0.15), cy + int(S * 0.26)],
                                radius=14, outline=fg, width=lw)
            for offset, length in [(-0.13, 0.22), (-0.04, 0.27), (0.04, 0.25), (0.13, 0.19)]:
                fx = cx + int(S * offset)
                d.line([fx, cy - int(S * 0.02), fx, cy - int(S * length)], fill=fg, width=lw)

        elif "Pinch" in kind:
            p1, p2 = (int(S * 0.28), int(S * 0.28)), (int(S * 0.52), int(S * 0.52))
            d.line([p1[0], p1[1], p2[0], p2[1]], fill=fg, width=lw)
            d.line([p1[0], p1[1], p1[0] + 16, p1[1]], fill=fg, width=lw)
            d.line([p1[0], p1[1], p1[0], p1[1] + 16], fill=fg, width=lw)
            d.ellipse([int(S * 0.50), int(S * 0.50), int(S * 0.74), int(S * 0.74)], outline=fg, width=lw)

        else:
            d.rounded_rectangle([cx - int(S * 0.14), cy + int(S * 0.02), cx + int(S * 0.14), cy + int(S * 0.26)],
                                radius=12, outline=fg, width=lw)
            d.line([cx - int(S * 0.04), cy + int(S * 0.02), cx - int(S * 0.04), cy - int(S * 0.25)], fill=fg, width=lw)
            d.ellipse([cx - int(S * 0.08), cy - int(S * 0.29), cx, cy - int(S * 0.21)], fill=fg)

        return ImageTk.PhotoImage(img.resize((size, size), Image.Resampling.LANCZOS))


# ==============================================================================
# 8. STANDALONE DESKTOP GUI APPLICATION (WITH HOVER 'ⓘ' TOOLTIPS & CAMERA ON/OFF)
# ==============================================================================
class WaveDesktopApp:
    BG_MAIN = "#F8FAFC"
    BG_CARD = "#FFFFFF"
    BORDER_LIGHT = "#E2E8F0"
    BLUE_PRIMARY = "#136DEC"
    BLUE_LIGHT_BG = "#E8F1FD"
    GREEN_TEXT = "#10B981"
    GREEN_BG = "#DCFCE7"
    RED_TEXT = "#EF4444"
    RED_BG = "#FEE2E2"
    SLATE_900 = "#0F172A"
    SLATE_700 = "#334155"
    SLATE_500 = "#64748B"
    SLATE_400 = "#94A3B8"
    PILL_BG = "#F1F5F9"

    DESCRIPTIONS = {
        "Right Swipe": ("Next Slide — Advances slide deck", "→ Right Arrow", "→ ArrowRight"),
        "Left Swipe": ("Previous Slide — Returns to prior slide", "← Left Arrow", "← ArrowLeft"),
        "Open Palm": ("Blackout Screen — Press 'B' to blank/unblank projector", "⌨ Key B", "KeyB"),
        "Pinch In / Out": ("Zoom Slide — Spread (+) or Pinch (-) like a tablet", "⊕ Ctrl + / -", "Ctrl + / -"),
        "Index Finger Point": ("Laser Pointer — Toggles virtual laser ('L' key)", "✦ Key L", "KeyL")
    }

    def __init__(self, root):
        self.root = root
        self.root.title("WAVE - Real-Time Gesture Recognition and Control Module")
        self.root.geometry("1280x850")
        self.root.minsize(1150, 800)
        self.root.configure(bg=self.BG_MAIN)

        self.ipc_config = IPCConfigManager(CONFIG_FILE)
        self.config = self.ipc_config.state
        self.os_dispatcher = OSActionDispatcher()
        self.vision_engine = VisionGestureEngine(MODEL_FILE)
        self.zoom_controller = TabletPinchZoomController()

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cam_start_on = self.config.get("camera_enabled", True)
        self.cam_stream = ThreadedCameraStream(src=0, width=640, height=480, start_enabled=cam_start_on)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self._init_mediapipe_hands()

        self.last_processed_cam_id = -1
        self.frame_count = 0
        self.cached_landmarks = None
        self.cached_hand_score = 0.0
        self.fist_hold_start = 0.0
        self.FIST_HOLD_DURATION = 2.0
        self.last_fist_toggle_time = 0.0
        self.last_recognized_ui_time = 0.0
        self.cam_off_rendered = False

        self.direction_banner_text = "DIRECTION: READY"
        self.latency_ema = ExponentialMovingAverage(alpha=0.25, fast_alpha=0.50)
        self.live_conf_ema = ExponentialMovingAverage(alpha=0.35, fast_alpha=0.70)
        self.rebinding_gesture = None
        self.tooltips = []

        self.assets = {
            "logo": GraphicsFactory.create_wave_logo(34),
            "avatar": GraphicsFactory.create_avatar_icon(32),
            "toggle_on": GraphicsFactory.create_toggle_switch(42, 24, True),
            "toggle_off": GraphicsFactory.create_toggle_switch(42, 24, False),
            "icons": {
                g: GraphicsFactory.create_gesture_icon(g, 38)
                for g in ["Right Swipe", "Left Swipe", "Open Palm", "Pinch In / Out", "Index Finger Point"]
            }
        }

        self._build_navbar()
        self._build_views()
        self._bind_shortcuts()

        self._sync_camera_ui(cam_start_on)
        initial_tab = self.config.get("active_tab", "live")
        if initial_tab not in ("live", "settings"):
            initial_tab = "live"
        self.switch_tab(initial_tab)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def _init_mediapipe_hands(self):
        if hasattr(self, "hands") and self.hands is not None:
            try:
                self.hands.close()
            except Exception:
                pass
        complexity = int(self.config.get("model_complexity", 0))
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=complexity,
            min_detection_confidence=0.60,
            min_tracking_confidence=0.60
        )

    def _apply_clahe_low_light(self, bgr_frame):
        ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = self.clahe.apply(y)
        merged = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _create_info_badge(self, parent, tooltip_title, tooltip_body):
        """
        Creates a clean 'ⓘ' circular badge that pops up a detailed explanation
        card when hovered over.
        """
        badge = tk.Label(
            parent, text=" ⓘ ", font=("Segoe UI", 9, "bold"),
            fg=self.BLUE_PRIMARY, bg=self.BLUE_LIGHT_BG,
            padx=3, pady=0, cursor="question_arrow"
        )
        badge.pack(side="left", padx=(8, 0))
        self.tooltips.append(HoverTooltip(badge, tooltip_title, tooltip_body))
        return badge

    def _build_navbar(self):
        self.navbar = tk.Frame(self.root, bg=self.BG_CARD, height=64, highlightthickness=1, highlightbackground=self.BORDER_LIGHT)
        self.navbar.pack(side="top", fill="x")
        self.navbar.pack_propagate(False)

        brand_frame = tk.Frame(self.navbar, bg=self.BG_CARD)
        brand_frame.pack(side="left", padx=(24, 20), pady=12)

        logo_lbl = tk.Label(brand_frame, image=self.assets["logo"], bg=self.BG_CARD)
        logo_lbl.pack(side="left", padx=(0, 10))

        title_lbl = tk.Label(brand_frame, text="WAVE", font=("Segoe UI", 14, "bold"), fg=self.SLATE_900, bg=self.BG_CARD)
        title_lbl.pack(side="left")

        tab_outer = tk.Frame(self.navbar, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT, padx=3, pady=3)
        tab_outer.pack(side="left", pady=12)

        self.tab_live_btn = tk.Label(
            tab_outer, text="Live Presentation", font=("Segoe UI", 9, "bold"),
            padx=16, pady=5, cursor="hand2"
        )
        self.tab_live_btn.pack(side="left")
        self.tab_live_btn.bind("<Button-1>", lambda e: self.switch_tab("live"))

        self.tab_settings_btn = tk.Label(
            tab_outer, text="⚙ Settings & Keybinds", font=("Segoe UI", 9, "bold"),
            padx=16, pady=5, cursor="hand2"
        )
        self.tab_settings_btn.pack(side="left")
        self.tab_settings_btn.bind("<Button-1>", lambda e: self.switch_tab("settings"))

        right_frame = tk.Frame(self.navbar, bg=self.BG_CARD)
        right_frame.pack(side="right", padx=24, pady=12)

        avatar_lbl = tk.Label(right_frame, image=self.assets["avatar"], bg=self.BG_CARD, cursor="hand2")
        avatar_lbl.pack(side="right", padx=(14, 0))
        avatar_lbl.bind("<Button-1>", lambda e: self.switch_tab("settings"))

        self.engine_toggle_lbl = tk.Label(
            right_frame,
            image=self.assets["toggle_on"] if self.config.get("recognition_enabled", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.engine_toggle_lbl.pack(side="right", padx=(8, 0))
        self.engine_toggle_lbl.bind("<Button-1>", lambda e: self.toggle_engine())

        engine_txt = tk.Label(right_frame, text="ENGINE", font=("Segoe UI", 8, "bold"), fg=self.SLATE_500, bg=self.BG_CARD)
        engine_txt.pack(side="right", padx=(12, 0))

        sep = tk.Frame(right_frame, bg=self.BORDER_LIGHT, width=1, height=22)
        sep.pack(side="right", padx=12)

        # Interactive Camera ON/OFF Toggle Pill in Top Navbar
        self.cam_pill = tk.Frame(
            right_frame, bg="#F1F5F9", highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=4, cursor="hand2"
        )
        self.cam_pill.pack(side="right")
        self.cam_pill.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_dot = tk.Label(self.cam_pill, text="●", font=("Segoe UI", 8), fg=self.GREEN_TEXT, bg="#F1F5F9", cursor="hand2")
        self.cam_dot.pack(side="left", padx=(0, 4))
        self.cam_dot.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_txt = tk.Label(self.cam_pill, text="CAM ACTIVE", font=("Segoe UI", 8, "bold"), fg=self.SLATE_700, bg="#F1F5F9", cursor="hand2")
        self.cam_txt.pack(side="left")
        self.cam_txt.bind("<Button-1>", lambda e: self.toggle_camera())

    def _build_views(self):
        self.content_container = tk.Frame(self.root, bg=self.BG_MAIN)
        self.content_container.pack(side="top", fill="both", expand=True)

        self.live_view = tk.Frame(self.content_container, bg=self.BG_MAIN)
        self.settings_view = tk.Frame(self.content_container, bg=self.BG_MAIN)

        self._build_live_presentation_view()
        self._build_vertical_settings_view()

    def _build_live_presentation_view(self):
        wrapper = tk.Frame(self.live_view, bg=self.BG_MAIN)
        wrapper.pack(expand=True, fill="both", padx=110, pady=(18, 18))

        self.video_card = tk.Frame(
            wrapper, bg="#0F172A", highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            width=960, height=500
        )
        self.video_card.pack(side="top", fill="both", expand=True, pady=(0, 14))
        self.video_card.pack_propagate(False)

        self.video_label = tk.Label(self.video_card, bg="#0F172A", bd=0)
        self.video_label.pack(fill="both", expand=True)

        self.telemetry_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=18, pady=14
        )
        self.telemetry_card.pack(side="top", fill="x", pady=(0, 12))

        left_info = tk.Frame(self.telemetry_card, bg=self.BG_CARD)
        left_info.pack(side="left")

        self.live_gesture_icon_lbl = tk.Label(
            left_info, image=self.assets["icons"]["Right Swipe"], bg=self.BG_CARD
        )
        self.live_gesture_icon_lbl.pack(side="left", padx=(0, 14))

        text_col = tk.Frame(left_info, bg=self.BG_CARD)
        text_col.pack(side="left")

        top_row = tk.Frame(text_col, bg=self.BG_CARD)
        top_row.pack(side="top", anchor="w")

        self.live_gesture_title = tk.Label(
            top_row, text="Awaiting Gesture...", font=("Segoe UI", 14, "bold"), fg=self.SLATE_900, bg=self.BG_CARD
        )
        self.live_gesture_title.pack(side="left", padx=(0, 10))

        self.recognized_badge = tk.Label(
            top_row, text=" STANDBY ", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_500, bg=self.PILL_BG, padx=6, pady=2
        )
        self.recognized_badge.pack(side="left")

        bottom_row = tk.Frame(text_col, bg=self.BG_CARD)
        bottom_row.pack(side="top", anchor="w", pady=(4, 0))

        tk.Label(
            bottom_row, text="Last Dispatched: ", font=("Segoe UI", 9), fg=self.SLATE_500, bg=self.BG_CARD
        ).pack(side="left")

        self.dispatched_action_lbl = tk.Label(
            bottom_row, text="None", font=("Segoe UI", 9, "bold"), fg=self.BLUE_PRIMARY, bg=self.BG_CARD
        )
        self.dispatched_action_lbl.pack(side="left", padx=(0, 8))

        self.dispatched_key_badge = tk.Label(
            bottom_row, text="—", font=("Consolas", 8, "bold"),
            fg=self.SLATE_700, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=6, pady=1
        )
        self.dispatched_key_badge.pack(side="left")

        pills_frame = tk.Frame(self.telemetry_card, bg=self.BG_CARD)
        pills_frame.pack(side="right")

        self.pill_cam_dot, self.pill_cam_val = self._create_status_pill(
            pills_frame, "CAMERA:", "ACTIVE", self.GREEN_TEXT
        )
        self.pill_hand_dot, self.pill_hand_val = self._create_status_pill(
            pills_frame, "HAND:", "NONE", self.SLATE_400
        )
        self.pill_mode_dot, self.pill_mode_val = self._create_status_pill(
            pills_frame, "MODE:", "ON", self.GREEN_TEXT
        )

        self.footer_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=16, pady=10
        )
        self.footer_card.pack(side="top", fill="x")

        # Pause Engine Button
        self.pause_btn_frame = tk.Frame(
            self.footer_card, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=12, pady=6, cursor="hand2"
        )
        self.pause_btn_frame.pack(side="left")
        self.pause_btn_frame.bind("<Button-1>", lambda e: self.toggle_engine())

        self.pause_icon_lbl = tk.Label(
            self.pause_btn_frame, text="⏸", font=("Segoe UI", 9), fg=self.BLUE_PRIMARY, bg=self.PILL_BG, cursor="hand2"
        )
        self.pause_icon_lbl.pack(side="left", padx=(0, 6))
        self.pause_icon_lbl.bind("<Button-1>", lambda e: self.toggle_engine())

        self.pause_txt_lbl = tk.Label(
            self.pause_btn_frame, text="Pause Engine", font=("Segoe UI", 9, "bold"), fg=self.SLATE_700, bg=self.PILL_BG, cursor="hand2"
        )
        self.pause_txt_lbl.pack(side="left", padx=(0, 8))
        self.pause_txt_lbl.bind("<Button-1>", lambda e: self.toggle_engine())

        space_keycap = tk.Label(
            self.pause_btn_frame, text="Space", font=("Segoe UI", 8),
            fg=self.SLATE_500, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=5, pady=0, cursor="hand2"
        )
        space_keycap.pack(side="left")
        space_keycap.bind("<Button-1>", lambda e: self.toggle_engine())

        # Dedicated Turn Off / Turn On Camera Button in Footer
        self.cam_btn_frame = tk.Frame(
            self.footer_card, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=12, pady=6, cursor="hand2"
        )
        self.cam_btn_frame.pack(side="left", padx=(10, 0))
        self.cam_btn_frame.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_btn_icon = tk.Label(
            self.cam_btn_frame, text="📷", font=("Segoe UI", 9), fg=self.RED_TEXT, bg=self.PILL_BG, cursor="hand2"
        )
        self.cam_btn_icon.pack(side="left", padx=(0, 6))
        self.cam_btn_icon.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_btn_txt = tk.Label(
            self.cam_btn_frame, text="Turn Off Camera", font=("Segoe UI", 9, "bold"), fg=self.SLATE_700, bg=self.PILL_BG, cursor="hand2"
        )
        self.cam_btn_txt.pack(side="left", padx=(0, 8))
        self.cam_btn_txt.bind("<Button-1>", lambda e: self.toggle_camera())

        c_keycap = tk.Label(
            self.cam_btn_frame, text="C", font=("Segoe UI", 8),
            fg=self.SLATE_500, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=5, pady=0, cursor="hand2"
        )
        c_keycap.pack(side="left")
        c_keycap.bind("<Button-1>", lambda e: self.toggle_camera())

        tk.Frame(self.footer_card, bg=self.BORDER_LIGHT, width=1, height=20).pack(side="left", padx=14)

        tk.Label(
            self.footer_card, text="LIVE CONFIDENCE: ", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        ).pack(side="left")

        self.confidence_val_lbl = tk.Label(
            self.footer_card, text="0.0%", font=("Segoe UI", 9, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        )
        self.confidence_val_lbl.pack(side="left")

        self.fps_footer_lbl = tk.Label(
            self.footer_card,
            text="CAMERA FPS: --   |   PIPELINE LATENCY: -- ms   |   ZOOM: IDLE",
            font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        )
        self.fps_footer_lbl.pack(side="right")

    def _create_status_pill(self, parent, prefix, value, dot_color):
        pill = tk.Frame(
            parent, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=4
        )
        pill.pack(side="left", padx=(8, 0))
        dot = tk.Label(pill, text="●", font=("Segoe UI", 7), fg=dot_color, bg=self.PILL_BG)
        dot.pack(side="left", padx=(0, 4))
        tk.Label(pill, text=prefix, font=("Segoe UI", 8, "bold"), fg=self.SLATE_500, bg=self.PILL_BG).pack(side="left")
        val = tk.Label(pill, text=f" {value}", font=("Segoe UI", 8, "bold"), fg=self.SLATE_900, bg=self.PILL_BG)
        val.pack(side="left")
        return dot, val

    # ==========================================================================
    # VERTICAL SETTINGS & KEYBINDS VIEW (WITH HOVER 'ⓘ' INFO TOOLTIPS)
    # ==========================================================================
    def _build_vertical_settings_view(self):
        wrapper = tk.Frame(self.settings_view, bg=self.BG_MAIN)
        wrapper.pack(fill="both", expand=True, padx=56, pady=16)

        header_row = tk.Frame(wrapper, bg=self.BG_MAIN)
        header_row.pack(fill="x", pady=(0, 12))

        title_col = tk.Frame(header_row, bg=self.BG_MAIN)
        title_col.pack(side="left", anchor="w")

        tk.Label(
            title_col, text="⚙  SYSTEM CONFIGURATION & HARDWARE ABSTRACTION LAYER", font=("Segoe UI", 8, "bold"),
            fg=self.BLUE_PRIMARY, bg=self.BG_MAIN
        ).pack(anchor="w")

        tk.Label(
            title_col, text="Settings & Gesture Key Mappings", font=("Segoe UI", 19, "bold"),
            fg=self.SLATE_900, bg=self.BG_MAIN
        ).pack(anchor="w")

        right_actions = tk.Frame(header_row, bg=self.BG_MAIN)
        right_actions.pack(side="right", anchor="s")

        reset_btn = tk.Label(
            right_actions, text="↺ Reset Defaults", font=("Segoe UI", 9, "bold"),
            fg="#FFFFFF", bg=self.BLUE_PRIMARY, padx=14, pady=6, cursor="hand2"
        )
        reset_btn.pack(side="right", padx=(8, 0))
        reset_btn.bind("<Button-1>", lambda e: self.reset_default_mappings())

        export_btn = tk.Label(
            right_actions, text="↓ Export JSON", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_700, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=12, pady=6, cursor="hand2"
        )
        export_btn.pack(side="right")
        export_btn.bind("<Button-1>", lambda e: self.export_json_dialog())

        # TOP VERTICAL CARD: Vision Engine & Camera Settings (With Hover 'ⓘ' Badges)
        video_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=22, pady=12
        )
        video_card.pack(fill="x", pady=(0, 14))

        vc_head = tk.Frame(video_card, bg=self.BG_CARD)
        vc_head.pack(fill="x", pady=(0, 6))
        tk.Label(vc_head, text="VISION ENGINE & HARDWARE CAMERA CONTROLS", font=("Segoe UI", 9, "bold"), fg=self.BLUE_PRIMARY, bg=self.BG_CARD).pack(side="left")

        self.live_lat_badge = tk.Label(vc_head, text="Pipeline Latency: -- ms", font=("Consolas", 8, "bold"), fg=self.GREEN_TEXT, bg=self.GREEN_BG, padx=8, pady=2)
        self.live_lat_badge.pack(side="right")

        # Row 0: Webcam Hardware Power (ON / OFF)
        v_row0 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row0.pack(fill="x", pady=4)
        tk.Label(v_row0, text="Webcam Sensor Capture (Hardware Power)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row0,
            "Webcam Sensor Capture (Shortcut: C)",
            "Physically turns your laptop webcam ON or OFF.\n"
            "• ON: Streams live frames into local RAM for gesture recognition.\n"
            "• OFF: Calls cap.release() to power down the camera hardware and turn off your webcam LED indicator."
        )
        self.settings_cam_toggle = tk.Label(
            v_row0,
            image=self.assets["toggle_on"] if self.config.get("camera_enabled", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.settings_cam_toggle.pack(side="right")
        self.settings_cam_toggle.bind("<Button-1>", lambda e: self.toggle_camera())

        # Row 1: AI Tracking Model Complexity (Lite vs Full)
        v_row1 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row1.pack(fill="x", pady=4)
        tk.Label(v_row1, text="AI Tracking Model Complexity", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row1,
            "MediaPipe TFLite Model Complexity (Lite vs. Full)",
            "Switches between Google's two neural network graphs:\n"
            "• Lite (Model 0): Uses hand_landmark_lite.tflite. Fastest CPU inference (~5–8 ms), ideal for standard or low-end laptops.\n"
            "• Full (Model 1): Uses hand_landmark_full.tflite. Deeper 3D skeletal regression for difficult hand angles at 3m–4.5m distance."
        )

        mp_btn_box = tk.Frame(v_row1, bg=self.BG_CARD)
        mp_btn_box.pack(side="right")
        curr_comp = int(self.config.get("model_complexity", 0))

        self.btn_comp_lite = tk.Label(
            mp_btn_box, text="Lite (Fastest Latency)", font=("Segoe UI", 8, "bold"),
            bg=self.BLUE_PRIMARY if curr_comp == 0 else self.PILL_BG,
            fg="#FFFFFF" if curr_comp == 0 else self.SLATE_700,
            padx=14, pady=4, cursor="hand2"
        )
        self.btn_comp_lite.pack(side="left", padx=(0, 6))
        self.btn_comp_lite.bind("<Button-1>", lambda e: self._set_model_complexity(0))

        self.btn_comp_full = tk.Label(
            mp_btn_box, text="Full (Deep Range)", font=("Segoe UI", 8, "bold"),
            bg=self.BLUE_PRIMARY if curr_comp == 1 else self.PILL_BG,
            fg="#FFFFFF" if curr_comp == 1 else self.SLATE_700,
            padx=14, pady=4, cursor="hand2"
        )
        self.btn_comp_full.pack(side="left")
        self.btn_comp_full.bind("<Button-1>", lambda e: self._set_model_complexity(1))

        # Row 2: Low-Light Contrast Boost (CLAHE)
        v_row2 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row2.pack(fill="x", pady=4)
        tk.Label(v_row2, text="Low-Light Contrast Boost (CLAHE)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row2,
            "Low-Light Contrast Boost (OpenCV YCrCb CLAHE)",
            "Applies Contrast Limited Adaptive Histogram Equalization (CLAHE) to the luminance (Y) channel before hand detection.\n"
            "• Boosts finger edge visibility in dim lecture halls or dark rooms without over-amplifying webcam sensor noise."
        )
        self.clahe_toggle = tk.Label(
            v_row2,
            image=self.assets["toggle_on"] if self.config.get("low_light_clahe", False) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.clahe_toggle.pack(side="right")
        self.clahe_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("low_light_clahe", self.clahe_toggle))

        # Row 3: Hand Skeleton Overlay Toggle
        v_row3 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row3.pack(fill="x", pady=4)
        tk.Label(v_row3, text="Hand Skeleton Overlay", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row3,
            "21-Point Hand Skeleton & Vector Overlay",
            "Renders the real-time 21-point MediaPipe knuckle landmarks, directional swipe vector arrow, and pinch-to-zoom HUD on the live video feed.\n"
            "• Turning this OFF hides the visual lines while keeping background gesture recognition active."
        )
        self.skel_toggle = tk.Label(
            v_row3,
            image=self.assets["toggle_on"] if self.config.get("show_skeleton_overlay", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.skel_toggle.pack(side="right")
        self.skel_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("show_skeleton_overlay", self.skel_toggle))

        # BOTTOM VERTICAL SECTION: Gesture Key Mappings Table
        col_header = tk.Frame(wrapper, bg=self.BG_MAIN)
        col_header.pack(fill="x", padx=12, pady=(2, 6))
        tk.Label(col_header, text="GESTURE KEYBINDINGS & ACTIVE STATES", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="left")
        tk.Label(col_header, text="STATE", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="right", padx=(0, 8))
        tk.Label(col_header, text="EMULATED SHORTCUT", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="right", padx=(0, 110))

        self.rows_container = tk.Frame(wrapper, bg=self.BG_MAIN)
        self.rows_container.pack(fill="both", expand=True)

        self.row_widgets = {}
        ordered_gestures = ["Right Swipe", "Left Swipe", "Open Palm", "Pinch In / Out", "Index Finger Point"]

        for gesture in ordered_gestures:
            row_card = tk.Frame(
                self.rows_container, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
                padx=16, pady=7
            )
            row_card.pack(fill="x", pady=(0, 7))

            tk.Label(row_card, image=self.assets["icons"][gesture], bg=self.BG_CARD).pack(side="left", padx=(0, 14))

            title_sub_col = tk.Frame(row_card, bg=self.BG_CARD)
            title_sub_col.pack(side="left")
            tk.Label(title_sub_col, text=gesture, font=("Segoe UI", 10, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(anchor="w")
            desc_text, default_pill, _ = self.DESCRIPTIONS.get(gesture, ("Action", "Key", "Key"))
            tk.Label(title_sub_col, text=desc_text, font=("Segoe UI", 8), fg=self.SLATE_500, bg=self.BG_CARD).pack(anchor="w")

            is_enabled = self.config.get("gesture_states", {}).get(gesture, gesture != "Open Palm")
            toggle_lbl = tk.Label(
                row_card,
                image=self.assets["toggle_on"] if is_enabled else self.assets["toggle_off"],
                bg=self.BG_CARD, cursor="hand2"
            )
            toggle_lbl.pack(side="right", padx=(12, 0))
            toggle_lbl.bind("<Button-1>", lambda e, g=gesture: self.toggle_gesture_row(g))

            rebind_btn = tk.Label(
                row_card, text="REBIND", font=("Segoe UI", 8, "bold"),
                fg=self.SLATE_700, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
                padx=10, pady=4, cursor="hand2"
            )
            rebind_btn.pack(side="right", padx=(14, 4))
            rebind_btn.bind("<Button-1>", lambda e, g=gesture: self.start_rebind(g))

            mapped_val = self.config.get("gesture_mappings", {}).get(gesture, default_pill)
            display_pill = self._format_keycode_badge(gesture, mapped_val)
            key_badge = tk.Label(
                row_card, text=display_pill, font=("Segoe UI", 9, "bold"),
                fg=self.BLUE_PRIMARY, bg=self.BLUE_LIGHT_BG, padx=12, pady=4
            )
            key_badge.pack(side="right")

            self.row_widgets[gesture] = {
                "badge": key_badge,
                "rebind": rebind_btn,
                "toggle": toggle_lbl
            }

    def toggle_camera(self):
        new_cam_state = not self.config.get("camera_enabled", True)
        self.config["camera_enabled"] = new_cam_state
        self.ipc_config.save(self.config)
        self.cam_stream.set_camera_enabled(new_cam_state)
        self.cached_landmarks = None
        self.cached_hand_score = 0.0
        self.zoom_controller.reset()
        self.vision_engine.reset_tracking()
        self.cam_off_rendered = False
        self._sync_camera_ui(new_cam_state)

    def _sync_camera_ui(self, cam_on):
        if hasattr(self, "settings_cam_toggle"):
            self.settings_cam_toggle.configure(image=self.assets["toggle_on"] if cam_on else self.assets["toggle_off"])
        if cam_on:
            self.cam_dot.configure(fg=self.GREEN_TEXT)
            self.cam_txt.configure(text="CAM ACTIVE", fg=self.SLATE_700)
            self.pill_cam_dot.configure(fg=self.GREEN_TEXT)
            self.pill_cam_val.configure(text=" ACTIVE")
            self.cam_btn_txt.configure(text="Turn Off Camera")
            self.cam_btn_icon.configure(fg=self.RED_TEXT)
        else:
            self.cam_dot.configure(fg=self.RED_TEXT)
            self.cam_txt.configure(text="CAM OFF (CLICK TO START)", fg=self.RED_TEXT)
            self.pill_cam_dot.configure(fg=self.RED_TEXT)
            self.pill_cam_val.configure(text=" OFF")
            self.pill_hand_dot.configure(fg=self.SLATE_400)
            self.pill_hand_val.configure(text=" NONE")
            self.cam_btn_txt.configure(text="Turn On Camera")
            self.cam_btn_icon.configure(fg=self.GREEN_TEXT)
            self.confidence_val_lbl.configure(text="0.0%", fg=self.SLATE_400)
            self.recognized_badge.configure(text=" STANDBY ", fg=self.SLATE_500, bg=self.PILL_BG)
            self.fps_footer_lbl.configure(text="CAMERA: OFF   |   PIPELINE LATENCY: 0.0 ms   |   ZOOM: IDLE")
            self.live_lat_badge.configure(text="Camera Off (0.0 ms)")

    def _render_camera_off_placeholder(self):
        card_w = max(self.video_card.winfo_width(), 860)
        card_h = max(self.video_card.winfo_height(), 460)
        canvas = np.full((card_h, card_w, 3), (38, 23, 15), dtype=np.uint8)
        cx, cy = card_w // 2, card_h // 2
        cv2.circle(canvas, (cx, cy - 28), 38, (65, 51, 35), -1, cv2.LINE_AA)
        cv2.putText(canvas, "CAMERA TURNED OFF", (cx - 165, cy + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (248, 250, 252), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Click 'Turn On Camera' or press 'C' to resume live video feed", (cx - 245, cy + 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (148, 163, 184), 1, cv2.LINE_AA)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        self.tk_video_img = ImageTk.PhotoImage(image=Image.fromarray(rgb))
        self.video_label.configure(image=self.tk_video_img)
        self.cam_off_rendered = True

    def _set_model_complexity(self, comp):
        self.config["model_complexity"] = int(comp)
        self.ipc_config.save(self.config)
        self.btn_comp_lite.configure(
            bg=self.BLUE_PRIMARY if comp == 0 else self.PILL_BG,
            fg="#FFFFFF" if comp == 0 else self.SLATE_700
        )
        self.btn_comp_full.configure(
            bg=self.BLUE_PRIMARY if comp == 1 else self.PILL_BG,
            fg="#FFFFFF" if comp == 1 else self.SLATE_700
        )
        self._init_mediapipe_hands()

    def _toggle_boolean_setting(self, key, label_widget):
        new_val = not self.config.get(key, True)
        self.config[key] = new_val
        self.ipc_config.save(self.config)
        label_widget.configure(image=self.assets["toggle_on"] if new_val else self.assets["toggle_off"])

    @staticmethod
    def _format_keycode_badge(gesture, mapped_val):
        low = str(mapped_val).lower()
        if "right" in low:
            return "→ Right Arrow"
        elif "left" in low:
            return "← Left Arrow"
        elif "ctrl" in low:
            return "⊕ Ctrl + / -"
        elif "key b" in low or low == "b":
            return "⌨ Key B"
        elif "key l" in low or low == "l":
            return "✦ Key L"
        return f"⌨ {mapped_val}"

    def _bind_shortcuts(self):
        self.root.bind("<Tab>", self._on_tab_key)
        self.root.bind("<space>", self._on_space_key)
        self.root.bind("<Key>", self._on_any_key)

    def _on_tab_key(self, event):
        new_tab = "settings" if self.config.get("active_tab", "live") == "live" else "live"
        self.switch_tab(new_tab)
        return "break"

    def _on_space_key(self, event):
        if self.rebinding_gesture is not None:
            return
        self.toggle_engine()

    def _on_any_key(self, event):
        if self.rebinding_gesture is not None:
            g = self.rebinding_gesture
            key_sym = event.keysym
            if key_sym == "Escape":
                self.rebinding_gesture = None
                self.row_widgets[g]["rebind"].configure(text="REBIND", bg=self.BG_CARD, fg=self.SLATE_700)
                return

            if key_sym in ("Right", "Left"):
                new_binding = f"{key_sym} Arrow"
            elif len(key_sym) == 1:
                new_binding = f"Key {key_sym.upper()}"
            else:
                new_binding = key_sym

            self.config["gesture_mappings"][g] = new_binding
            self.ipc_config.save(self.config)
            self.row_widgets[g]["badge"].configure(text=self._format_keycode_badge(g, new_binding))
            self.row_widgets[g]["rebind"].configure(text="REBIND", bg=self.BG_CARD, fg=self.SLATE_700)
            self.rebinding_gesture = None
            return

        if event.keysym.lower() == 'p':
            self.toggle_engine()
        elif event.keysym.lower() == 'c':
            self.toggle_camera()
        elif event.keysym.lower() == 'q' or event.keysym == 'Escape':
            self.on_close()

    def switch_tab(self, tab_name):
        if tab_name not in ("live", "settings"):
            tab_name = "live"
        self.config["active_tab"] = tab_name
        self.ipc_config.save(self.config)

        if tab_name == "live":
            self.settings_view.pack_forget()
            self.live_view.pack(fill="both", expand=True)
            self.tab_live_btn.configure(bg=self.BG_CARD, fg=self.SLATE_900)
            self.tab_settings_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
            if not self.config.get("camera_enabled", True):
                self.cam_off_rendered = False
        else:
            self.live_view.pack_forget()
            self.settings_view.pack(fill="both", expand=True)
            self.tab_settings_btn.configure(bg=self.BG_CARD, fg=self.BLUE_PRIMARY)
            self.tab_live_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)

    def toggle_engine(self):
        new_state = not self.config.get("recognition_enabled", True)
        self.config["recognition_enabled"] = new_state
        self.ipc_config.save(self.config)
        self.zoom_controller.reset()
        self.vision_engine.reset_tracking()
        self._sync_engine_ui(new_state)

    def _sync_engine_ui(self, is_on):
        self.engine_toggle_lbl.configure(image=self.assets["toggle_on"] if is_on else self.assets["toggle_off"])
        if is_on:
            self.pill_mode_dot.configure(fg=self.GREEN_TEXT)
            self.pill_mode_val.configure(text=" ON")
            self.pause_txt_lbl.configure(text="Pause Engine")
            self.pause_icon_lbl.configure(text="⏸", fg=self.BLUE_PRIMARY)
        else:
            self.pill_mode_dot.configure(fg="#EF4444")
            self.pill_mode_val.configure(text=" OFF")
            self.pause_txt_lbl.configure(text="Resume Engine")
            self.pause_icon_lbl.configure(text="▶", fg=self.GREEN_TEXT)

    def toggle_gesture_row(self, gesture):
        states = self.config.setdefault("gesture_states", {})
        new_val = not states.get(gesture, True)
        states[gesture] = new_val
        self.ipc_config.save(self.config)
        self.row_widgets[gesture]["toggle"].configure(
            image=self.assets["toggle_on"] if new_val else self.assets["toggle_off"]
        )

    def start_rebind(self, gesture):
        if self.rebinding_gesture and self.rebinding_gesture in self.row_widgets:
            self.row_widgets[self.rebinding_gesture]["rebind"].configure(text="REBIND", bg=self.BG_CARD, fg=self.SLATE_700)
        self.rebinding_gesture = gesture
        self.row_widgets[gesture]["rebind"].configure(text="PRESS KEY...", bg=self.BLUE_LIGHT_BG, fg=self.BLUE_PRIMARY)

    def export_json_dialog(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Configuration", "*.json")],
            initialfile="wave_mappings_export.json"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("WAVE Export", f"Configuration exported successfully to:\n{path}")

    def reset_default_mappings(self):
        self.config["gesture_mappings"] = IPCConfigManager.DEFAULT_CONFIG["gesture_mappings"].copy()
        self.config["gesture_states"] = IPCConfigManager.DEFAULT_CONFIG["gesture_states"].copy()
        self.ipc_config.save(self.config)
        for g, widgets in self.row_widgets.items():
            widgets["badge"].configure(text=self._format_keycode_badge(g, self.config["gesture_mappings"][g]))
            is_on = self.config["gesture_states"].get(g, True)
            widgets["toggle"].configure(image=self.assets["toggle_on"] if is_on else self.assets["toggle_off"])

    def _draw_pinch_hud(self, img, pt_thumb, pt_index, zoom_state, smoothed_ratio):
        mid_x = (pt_thumb[0] + pt_index[0]) // 2
        mid_y = (pt_thumb[1] + pt_index[1]) // 2

        if zoom_state == TabletPinchZoomController.STATE_ZOOM_IN:
            line_color = (16, 185, 129)
            tag = "PINCH OUT (ZOOM IN +)"
        elif zoom_state == TabletPinchZoomController.STATE_ZOOM_OUT:
            line_color = (236, 109, 19)
            tag = "PINCH IN (ZOOM OUT -)"
        else:
            line_color = (0, 180, 255)
            tag = "PINCH LOCKED"

        cv2.line(img, pt_thumb, pt_index, line_color, 4, cv2.LINE_AA)
        cv2.circle(img, pt_thumb, 9, line_color, -1, cv2.LINE_AA)
        cv2.circle(img, pt_index, 9, line_color, -1, cv2.LINE_AA)
        cv2.circle(img, (mid_x, mid_y), 6, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(img, tag, (mid_x + 14, mid_y + 5),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, line_color, 2, cv2.LINE_AA)

    def _update_telemetry_card(self, gesture_name, confidence_pct, custom_action=None, custom_key=None):
        icon_key = gesture_name if gesture_name in self.assets["icons"] else "Pinch In / Out"
        self.live_gesture_icon_lbl.configure(image=self.assets["icons"][icon_key])
        self.live_gesture_title.configure(text=gesture_name)
        self.recognized_badge.configure(text=" RECOGNIZED ", fg=self.GREEN_TEXT, bg=self.GREEN_BG)
        self.last_recognized_ui_time = time.time()

        desc, _, key_tag = self.DESCRIPTIONS.get(gesture_name, (custom_action or "Zoom Slide", "", "Ctrl + / -"))
        short_desc = desc.split("—")[0].strip()
        self.dispatched_action_lbl.configure(text=custom_action if custom_action else short_desc)
        self.dispatched_key_badge.configure(text=custom_key if custom_key else key_tag)
        self.confidence_val_lbl.configure(text=f"{confidence_pct:.1f}%", fg=self.GREEN_TEXT)

    def update_loop(self):
        cam_on = self.config.get("camera_enabled", True)
        is_live_tab = (self.config.get("active_tab", "live") == "live")

        if not cam_on:
            if is_live_tab and not self.cam_off_rendered:
                self._render_camera_off_placeholder()
            self.root.after(30, self.update_loop)
            return

        ret, raw_frame, cam_frame_id, measured_cam_fps = self.cam_stream.read()

        if ret and raw_frame is not None and cam_frame_id != self.last_processed_cam_id:
            self.cam_off_rendered = False
            self.last_processed_cam_id = cam_frame_id
            self.frame_count += 1
            t_pipeline_start = cv2.getTickCount()
            freq = cv2.getTickFrequency()

            recognition_on = self.config.get("recognition_enabled", True)
            self._sync_engine_ui(recognition_on)

            frame_skip = max(1, int(self.config.get("frame_skip", 2)))
            should_run_mp = (self.cached_landmarks is not None) or (self.frame_count % frame_skip == 0)

            if should_run_mp:
                infer_small = cv2.resize(raw_frame, (480, 360), interpolation=cv2.INTER_NEAREST)
                if self.config.get("low_light_clahe", False):
                    infer_small = self._apply_clahe_low_light(infer_small)
                rgb_small = cv2.cvtColor(infer_small, cv2.COLOR_BGR2RGB)
                rgb_small.flags.writeable = False
                results = self.hands.process(rgb_small)

                if results.multi_hand_landmarks:
                    self.cached_landmarks = results.multi_hand_landmarks[0]
                    if results.multi_handedness:
                        self.cached_hand_score = float(results.multi_handedness[0].classification[0].score)
                    else:
                        self.cached_hand_score = 0.92
                else:
                    self.cached_landmarks = None
                    self.cached_hand_score = 0.0

            hand_detected = self.cached_landmarks is not None
            now_time = time.time()
            swipe_cooldown = float(self.config.get("swipe_cooldown", 0.42))
            zoom_sens = float(self.config.get("zoom_sensitivity", 0.050))

            if is_live_tab:
                card_w = max(self.video_card.winfo_width(), 860)
                card_h = max(self.video_card.winfo_height(), 460)
                base_vis = self._apply_clahe_low_light(raw_frame) if self.config.get("low_light_clahe", False) else raw_frame
                display_frame = cv2.resize(base_vis, (card_w, card_h), interpolation=cv2.INTER_NEAREST)
            else:
                card_w, card_h = 960, 500
                display_frame = None

            if not hand_detected:
                self.pill_hand_dot.configure(fg=self.SLATE_400)
                self.pill_hand_val.configure(text=" NONE")
                self.direction_banner_text = "DIRECTION: READY (NO HAND)"
                self.live_conf_ema.reset()
                self.confidence_val_lbl.configure(text="0.0%", fg=self.SLATE_400)
                self.recognized_badge.configure(text=" STANDBY ", fg=self.SLATE_500, bg=self.PILL_BG)
                self.zoom_controller.reset()
                self.vision_engine.reset_tracking()
                self.fist_hold_start = 0.0
            else:
                self.pill_hand_dot.configure(fg=self.BLUE_PRIMARY)
                self.pill_hand_val.configure(text=f" DETECTED ({self.cached_hand_score * 100:.0f}%)")

                wrist = self.cached_landmarks.landmark[0]
                thumb_tip = self.cached_landmarks.landmark[4]
                index_tip = self.cached_landmarks.landmark[8]

                if is_live_tab and self.config.get("show_skeleton_overlay", True):
                    self.mp_drawing.draw_landmarks(
                        display_frame,
                        self.cached_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing.DrawingSpec(color=(60, 255, 60), thickness=2, circle_radius=3),
                        self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                    )

                pt_wrist = (int(wrist.x * card_w), int(wrist.y * card_h))
                pt_index = (int(index_tip.x * card_w), int(index_tip.y * card_h))
                pt_thumb = (int(thumb_tip.x * card_w), int(thumb_tip.y * card_h))

                in_safe_margin = self.vision_engine.is_within_safe_frame_margin(self.cached_landmarks)

                gesture_states = self.config.get("gesture_states", {})
                norm_vel_x, velocity_swipe, vel_conf = self.vision_engine.compute_velocity_and_swipe(
                    self.cached_landmarks, now_time, mp_hand_score=self.cached_hand_score
                )
                static_dir_gesture, dir_banner, dir_conf = self.vision_engine.detect_static_direction_pose(
                    self.cached_landmarks, mp_hand_score=self.cached_hand_score
                )
                mlp_label, mlp_conf = self.vision_engine.classify_static_mlp(self.cached_landmarks, gesture_states)

                self.direction_banner_text = dir_banner if in_safe_margin else "DIRECTION: EDGE BORDER (SAFE GUARD)"

                if (now_time - self.last_recognized_ui_time) > 0.60 and self.zoom_controller.state == TabletPinchZoomController.STATE_IDLE:
                    self.recognized_badge.configure(text=" STANDBY ", fg=self.SLATE_500, bg=self.PILL_BG)

                # PRIORITY 1: Fast Dynamic Swipe OR Deliberate Left/Right Pointing
                if recognition_on and in_safe_margin and (velocity_swipe is not None or static_dir_gesture is not None):
                    self.zoom_controller.reset()
                    if is_live_tab:
                        cv2.arrowedLine(display_frame, pt_wrist, pt_index, (0, 0, 255), 4, cv2.LINE_AA, 0, 0.15)

                    detected_gesture = velocity_swipe if velocity_swipe is not None else static_dir_gesture
                    is_deliberate_point = (static_dir_gesture is not None)
                    raw_conf_pct = (vel_conf if velocity_swipe is not None else dir_conf) * 100.0
                    dyn_conf_pct = self.live_conf_ema.filter(raw_conf_pct)
                    self.confidence_val_lbl.configure(text=f"{dyn_conf_pct:.1f}%", fg=self.GREEN_TEXT)

                    recoil_blocked = self.vision_engine.is_recoil_blocked(
                        detected_gesture, is_deliberate_point, now_time
                    )

                    if not recoil_blocked and (now_time - self.vision_engine.last_trigger_time) > swipe_cooldown:
                        action_res = self.os_dispatcher.execute_shortcut(
                            detected_gesture,
                            self.config.get("gesture_mappings", {}),
                            gesture_states
                        )
                        if action_res != "Disabled":
                            self._update_telemetry_card(detected_gesture, dyn_conf_pct)
                            self.vision_engine.last_trigger_time = now_time
                            self.vision_engine.last_swipe_dir = detected_gesture
                            self.vision_engine.last_swipe_time = now_time
                            self.vision_engine.reset_tracking()

                # PRIORITY 2: Tablet Pinch-to-Zoom OR Other Static Poses
                elif recognition_on and in_safe_margin:
                    zoom_state = self.zoom_controller.state
                    zoom_ticks = 0
                    smoothed_ratio = 0.0
                    zoom_label = ""
                    zoom_conf_pct = 0.0

                    if self.config.get("enable_zoom_optional", True) and gesture_states.get("Pinch In / Out", True):
                        zoom_state, zoom_ticks, smoothed_ratio, zoom_label, zoom_conf_pct = self.zoom_controller.update(
                            self.cached_landmarks, norm_vel_x,
                            mp_hand_score=self.cached_hand_score,
                            step_sensitivity=zoom_sens
                        )

                    if zoom_state != TabletPinchZoomController.STATE_IDLE:
                        if is_live_tab:
                            self._draw_pinch_hud(display_frame, pt_thumb, pt_index, zoom_state, smoothed_ratio)
                        self.direction_banner_text = zoom_label
                        dyn_zoom_conf = self.live_conf_ema.filter(zoom_conf_pct)
                        self.confidence_val_lbl.configure(text=f"{dyn_zoom_conf:.1f}%", fg=self.GREEN_TEXT)

                        if zoom_state == TabletPinchZoomController.STATE_ARMED and zoom_ticks == 0:
                            self.recognized_badge.configure(text=" PINCH LOCKED ", fg=self.BLUE_PRIMARY, bg=self.BLUE_LIGHT_BG)

                        if zoom_ticks > 0:
                            self.os_dispatcher.execute_zoom(zoom_ticks)
                            self._update_telemetry_card(
                                "Pinch Out (Zoom In)", dyn_zoom_conf,
                                custom_action="Zoom In (+)", custom_key="Ctrl + Plus (+)"
                            )
                        elif zoom_ticks < 0:
                            self.os_dispatcher.execute_zoom(zoom_ticks)
                            self._update_telemetry_card(
                                "Pinch In (Zoom Out)", dyn_zoom_conf,
                                custom_action="Zoom Out (-)", custom_key="Ctrl + Minus (-)"
                            )
                        self.vision_engine.prediction_buffer.clear()
                    else:
                        if is_live_tab:
                            cv2.arrowedLine(display_frame, pt_wrist, pt_index, (0, 0, 255), 4, cv2.LINE_AA, 0, 0.15)
                        live_idle_conf = self.live_conf_ema.filter(mlp_conf * 100.0)
                        self.confidence_val_lbl.configure(text=f"{live_idle_conf:.1f}%", fg=self.SLATE_700)

                        if mlp_label not in ("Neutral", "Pinch In / Out"):
                            self.vision_engine.prediction_buffer.append(mlp_label)
                            if (now_time - self.vision_engine.last_trigger_time) > swipe_cooldown:
                                if self.vision_engine.prediction_buffer.count(mlp_label) >= 2:
                                    action_res = self.os_dispatcher.execute_shortcut(
                                        mlp_label,
                                        self.config.get("gesture_mappings", {}),
                                        gesture_states
                                    )
                                    if action_res != "Disabled":
                                        self._update_telemetry_card(mlp_label, live_idle_conf)
                                        self.vision_engine.last_trigger_time = now_time
                                        self.vision_engine.reset_tracking()

                is_fist = self.vision_engine.is_strict_closed_fist(
                    self.cached_landmarks,
                    zoom_active=(self.zoom_controller.state != TabletPinchZoomController.STATE_IDLE)
                )
                if is_fist and (now_time - self.last_fist_toggle_time) > 2.5:
                    if self.fist_hold_start == 0.0:
                        self.fist_hold_start = now_time
                    else:
                        held_sec = now_time - self.fist_hold_start
                        if held_sec >= 0.4:
                            rem = max(0.0, self.FIST_HOLD_DURATION - held_sec)
                            self.direction_banner_text = f"HOLD FIST TO PAUSE ({rem:.1f}s)"
                        if held_sec >= self.FIST_HOLD_DURATION:
                            self.toggle_engine()
                            recognition_on = self.config.get("recognition_enabled", True)
                            self.last_fist_toggle_time = now_time
                            self.fist_hold_start = 0.0
                else:
                    self.fist_hold_start = 0.0

            pipeline_ms = ((cv2.getTickCount() - t_pipeline_start) / freq) * 1000.0
            mean_lat = self.latency_ema.filter(pipeline_ms)

            self.cam_txt.configure(text=f"CAM ACTIVE ({measured_cam_fps:.1f} FPS)")
            self.fps_footer_lbl.configure(
                text=f"CAMERA FPS: {measured_cam_fps:.1f} FPS   |   PIPELINE LATENCY: {mean_lat:.1f} ms   |   ZOOM: {self.zoom_controller.state}"
            )
            self.live_lat_badge.configure(text=f"Pipeline Latency: {mean_lat:.1f} ms")

            if is_live_tab and display_frame is not None:
                overlay_text = self.direction_banner_text if recognition_on else "ENGINE PAUSED"
                overlay_color = (50, 235, 50) if recognition_on else (60, 60, 235)
                cv2.putText(
                    display_frame, overlay_text, (16, 36),
                    cv2.FONT_HERSHEY_DUPLEX, 1.05, overlay_color, 2, cv2.LINE_AA
                )
                rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_display)
                self.tk_video_img = ImageTk.PhotoImage(image=pil_img)
                self.video_label.configure(image=self.tk_video_img)

        self.root.after(4, self.update_loop)

    def on_close(self):
        try:
            self.cam_stream.release()
            self.hands.close()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WaveDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()