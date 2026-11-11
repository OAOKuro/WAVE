import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import time
import math
import os
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from collections import deque
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Controller as MouseController

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
# 1. MODULAR INTEGRATION INTERFACE (SHARED CONFIG & IPC STATE MANAGER)
# ==============================================================================
class IPCConfigManager:
    """
    Handles shared configuration and Inter-Process Communication (IPC) state
    via config.json so partner modules can read/write state seamlessly.
    """
    DEFAULT_CONFIG = {
        "recognition_enabled": True,
        "min_detection_confidence": 0.7,
        "min_tracking_confidence": 0.7,
        "frame_skip": 2,
        "swipe_cooldown": 0.6,
        "enable_zoom_optional": True,
        "enable_roi_optional": False,
        "active_tab": "live",  # 'live' or 'mapping'
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
            "Open Palm": True,
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
            if merged.get("min_detection_confidence", 0.7) < 0.7:
                merged["min_detection_confidence"] = 0.7
            if merged.get("frame_skip", 2) < 2:
                merged["frame_skip"] = 2
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
# 2. OS HARDWARE ABSTRACTION LAYER (KEYBOARD & ZOOM EMULATION)
# ==============================================================================
class OSActionDispatcher:
    """
    Decoupled OS command dispatcher. Translates high-level vision events into
    lag-free native OS keystrokes and smooth zoom commands.
    """
    def __init__(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()

    def execute_zoom(self, ticks):
        """
        Executes tablet-style OS zoom:
        ticks > 0 -> Zoom In (fingers pulled apart)
        ticks < 0 -> Zoom Out (fingers brought together)
        """
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

        return "Ctrl + Zoom In (+)" if ticks > 0 else "Ctrl + Zoom Out (-)"

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
# 3. SIGNAL PROCESSING: EXPONENTIAL MOVING AVERAGE (EMA)
# ==============================================================================
class ExponentialMovingAverage:
    """
    Cascaded Exponential Moving Average (EMA) filter with adaptive velocity boost
    for ultra-smooth, jitter-free pinch-to-zoom ratio tracking.
    """
    def __init__(self, alpha=0.32, fast_alpha=0.65, velocity_threshold=0.25):
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
# 4. TABLET-STYLE DEPTH-INVARIANT PINCH-TO-ZOOM STATE MACHINE
# ==============================================================================
class TabletPinchZoomController:
    """
    High-precision, depth-invariant tablet pinch-to-zoom controller.
    Strictly rejects sideways pointing gestures (static Left/Right swipes)
    so zoom only engages when the user intentionally pinches thumb & index tip-to-tip.
    """
    STATE_IDLE = "IDLE"
    STATE_ARMED = "ARMED"
    STATE_ZOOMING = "ZOOMING"

    def __init__(self):
        self.ratio_ema = ExponentialMovingAverage(alpha=0.30, fast_alpha=0.68)
        self.state = self.STATE_IDLE
        self.anchor_ratio = 0.0
        self.last_smoothed_ratio = 0.0
        self.zoom_accumulator = 0.0

        self.START_THRESHOLD = 0.34     # Tighter tip-to-tip touch required to arm pinch
        self.RELEASE_THRESHOLD = 1.55   # Open wider than this to release clutch
        self.DEADZONE_DELTA = 0.045     # Must intentionally spread/contract beyond this to zoom
        self.STEP_SENSITIVITY = 0.075   # Normalized ratio change per OS zoom tick
        self.HYSTERESIS = 0.016

        self.arm_start_time = 0.0
        self.ARM_HOLD_TIME = 0.12       # Hold pinch touch for 120ms to lock clutch
        self.last_zoom_dir = 0

    def reset(self):
        self.state = self.STATE_IDLE
        self.ratio_ema.reset()
        self.zoom_accumulator = 0.0
        self.arm_start_time = 0.0
        self.last_zoom_dir = 0

    @staticmethod
    def _dist(lm1, lm2):
        return math.hypot(lm1.x - lm2.x, lm1.y - lm2.y)

    def _is_pinch_posture_isolated(self, landmarks):
        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]
        middle_tip = landmarks.landmark[12]
        middle_pip = landmarks.landmark[10]
        ring_tip = landmarks.landmark[16]
        ring_pip = landmarks.landmark[14]
        pinky_tip = landmarks.landmark[20]
        pinky_pip = landmarks.landmark[18]

        # CRITICAL GUARD: If index finger is pointing strongly left or right,
        # the user is performing a directional point/swipe, NOT a pinch!
        dx_point = index_tip.x - wrist.x
        dy_point = index_tip.y - wrist.y
        if abs(dx_point) > 0.12 and abs(dx_point) > abs(dy_point) * 1.15:
            if self.state == self.STATE_IDLE:
                return False

        index_extended = self._dist(index_tip, wrist) > self._dist(index_mcp, wrist) * 0.90
        thumb_extended = self._dist(thumb_tip, wrist) > self._dist(index_mcp, wrist) * 0.72
        if not (index_extended and thumb_extended):
            return False

        if self.state in (self.STATE_ARMED, self.STATE_ZOOMING):
            return True

        ring_curled = self._dist(ring_tip, wrist) < self._dist(ring_pip, wrist) * 1.18
        pinky_curled = self._dist(pinky_tip, wrist) < self._dist(pinky_pip, wrist) * 1.18
        middle_curled = self._dist(middle_tip, wrist) < self._dist(middle_pip, wrist) * 1.20
        return (ring_curled and pinky_curled) or middle_curled

    def update(self, landmarks, wrist_velocity_x):
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

        if abs(wrist_velocity_x) > 0.35:
            self.reset()
            return self.STATE_IDLE, 0, smoothed_ratio, ""

        if not self._is_pinch_posture_isolated(landmarks):
            self.reset()
            return self.STATE_IDLE, 0, smoothed_ratio, ""

        zoom_ticks = 0
        status_label = ""

        if self.state == self.STATE_IDLE:
            if smoothed_ratio <= self.START_THRESHOLD:
                if self.arm_start_time == 0.0:
                    self.arm_start_time = now
                elif (now - self.arm_start_time) >= self.ARM_HOLD_TIME:
                    self.state = self.STATE_ARMED
                    self.anchor_ratio = smoothed_ratio
                    self.last_smoothed_ratio = smoothed_ratio
                    self.zoom_accumulator = 0.0
                    status_label = "PINCH LOCKED"
            else:
                self.arm_start_time = 0.0

        elif self.state == self.STATE_ARMED:
            status_label = "PINCH LOCKED"
            delta_from_anchor = smoothed_ratio - self.anchor_ratio

            # Only enter ZOOMING when spreading fingers apart (or pinching tighter) beyond deadzone
            if abs(delta_from_anchor) >= self.DEADZONE_DELTA:
                self.state = self.STATE_ZOOMING
                self.last_smoothed_ratio = smoothed_ratio
                self.zoom_accumulator = delta_from_anchor
            elif smoothed_ratio > self.RELEASE_THRESHOLD:
                self.reset()

        elif self.state == self.STATE_ZOOMING:
            if smoothed_ratio > self.RELEASE_THRESHOLD:
                self.reset()
                return self.STATE_IDLE, 0, smoothed_ratio, ""

            delta = smoothed_ratio - self.last_smoothed_ratio
            self.last_smoothed_ratio = smoothed_ratio

            if (self.last_zoom_dir > 0 and delta < 0 and abs(delta) < self.HYSTERESIS) or \
               (self.last_zoom_dir < 0 and delta > 0 and abs(delta) < self.HYSTERESIS):
                delta = 0.0

            self.zoom_accumulator += delta

            if abs(self.zoom_accumulator) >= self.STEP_SENSITIVITY:
                zoom_ticks = int(self.zoom_accumulator / self.STEP_SENSITIVITY)
                self.zoom_accumulator -= zoom_ticks * self.STEP_SENSITIVITY
                self.last_zoom_dir = 1 if zoom_ticks > 0 else -1

            zoom_pct = int(np.clip((smoothed_ratio / self.START_THRESHOLD) * 100, 50, 350))
            if self.last_zoom_dir >= 0:
                status_label = f"ZOOM IN (+{zoom_pct}%)"
            else:
                status_label = f"ZOOM OUT ({zoom_pct}%)"

        return self.state, zoom_ticks, smoothed_ratio, status_label


# ==============================================================================
# 5. HYBRID STATIC-DIRECTION & DYNAMIC-VELOCITY GESTURE ENGINE
# ==============================================================================
class VisionGestureEngine:
    """
    Recognizes BOTH:
    1. Dynamic multi-frame horizontal hand swipes (X-velocity).
    2. Static directional hand pointing poses (pointing Left or Right with the
       red vector arrow, as shown in Screenshot 2).
    """
    def __init__(self, model_path=MODEL_FILE):
        if not os.path.exists(model_path):
            train_mlp.main()

        checkpoint = torch.load(model_path)
        self.classes = checkpoint["classes"]
        self.model = train_mlp.WAVE_MLP(input_dim=42, num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

        self.trajectory = deque(maxlen=6)
        self.prediction_buffer = deque(maxlen=3)
        self.last_trigger_time = 0.0

    def reset_tracking(self):
        self.trajectory.clear()
        self.prediction_buffer.clear()

    @staticmethod
    def is_closed_fist(landmarks):
        wrist = landmarks.landmark[0]
        index_tip, index_mcp = landmarks.landmark[8], landmarks.landmark[5]
        middle_tip, middle_mcp = landmarks.landmark[12], landmarks.landmark[9]
        ring_tip, ring_mcp = landmarks.landmark[16], landmarks.landmark[13]
        pinky_tip, pinky_mcp = landmarks.landmark[20], landmarks.landmark[17]

        return (
            math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) < math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y) and
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) < math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y) and
            math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < math.hypot(ring_mcp.x - wrist.x, ring_mcp.y - wrist.y) and
            math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < math.hypot(pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y)
        )

    @staticmethod
    def detect_static_direction_pose(landmarks):
        """
        Detects when the user holds a deliberate directional pointing pose
        (red vector arrow pointing clearly Left or Right).
        """
        wrist = landmarks.landmark[0]
        index_tip = landmarks.landmark[8]
        index_mcp = landmarks.landmark[5]

        dx = index_tip.x - wrist.x
        dy = index_tip.y - wrist.y
        index_len = math.hypot(dx, dy)
        palm_len = math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y)

        # Ensure index finger is extended outward and pointing predominantly horizontally
        if index_len > palm_len * 1.15 and abs(dx) > abs(dy) * 1.10:
            if dx < -0.11:
                return "Left Swipe", "DIRECTION: LEFT"
            elif dx > 0.11:
                return "Right Swipe", "DIRECTION: RIGHT"

        if dx < -0.08:
            return None, "DIRECTION: LEFT"
        elif dx > 0.08:
            return None, "DIRECTION: RIGHT"
        return None, "DIRECTION: CENTER"

    def compute_velocity_and_swipe(self, wrist_x, now_time):
        self.trajectory.append((wrist_x, now_time))
        if len(self.trajectory) < 3:
            return 0.0, None

        x0, t0 = self.trajectory[0]
        x1, t1 = self.trajectory[-1]
        dt = max(t1 - t0, 1e-3)
        dx = x1 - x0
        velocity_x = dx / dt

        diffs = [self.trajectory[i][0] - self.trajectory[i - 1][0] for i in range(1, len(self.trajectory))]
        consistent_right = sum(1 for d in diffs if d > 0.004) >= len(diffs) - 1
        consistent_left = sum(1 for d in diffs if d < -0.004) >= len(diffs) - 1

        if dx > 0.065 and velocity_x > 0.35 and consistent_right:
            return velocity_x, "Right Swipe"
        elif dx < -0.065 and velocity_x < -0.35 and consistent_left:
            return velocity_x, "Left Swipe"

        return velocity_x, None

    def classify_static_mlp(self, landmarks):
        features_42 = train_mlp.extract_42_features(landmarks).reshape(1, -1)
        tensor_input = torch.tensor(features_42, dtype=torch.float32)
        with torch.no_grad():
            outputs = self.model(tensor_input)
            probs = torch.softmax(outputs, dim=1).numpy()[0]

        pred_idx = int(np.argmax(probs))
        label = self.classes[pred_idx]
        conf = float(probs[pred_idx])
        if "pinch" in label.lower():
            return "Neutral", 0.0
        return label, conf


# ==============================================================================
# 6. HIGH-DPI PIL ICON & GRAPHICS FACTORY (EXACT SCREENSHOT MATCH)
# ==============================================================================
class GraphicsFactory:
    """
    Renders 4x supersampled anti-aliased icons, rounded badges, toggle switches,
    and the custom WAVE hand-wave logo using PIL for native Tkinter display.
    """
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
    def create_gesture_icon(kind, size=44):
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
# 7. STANDALONE DESKTOP GUI APPLICATION (EXACT SCREENSHOT REPLICATION)
# ==============================================================================
class WaveDesktopApp:
    BG_MAIN = "#F8FAFC"
    BG_CARD = "#FFFFFF"
    BORDER_LIGHT = "#E2E8F0"
    BLUE_PRIMARY = "#136DEC"
    BLUE_LIGHT_BG = "#E8F1FD"
    GREEN_TEXT = "#10B981"
    GREEN_BG = "#DCFCE7"
    SLATE_900 = "#0F172A"
    SLATE_700 = "#334155"
    SLATE_500 = "#64748B"
    SLATE_400 = "#94A3B8"
    PILL_BG = "#F1F5F9"

    DESCRIPTIONS = {
        "Right Swipe": ("Next Slide", "→ Right Arrow", "→ ArrowRight"),
        "Left Swipe": ("Previous Slide", "← Left Arrow", "← ArrowLeft"),
        "Open Palm": ("Blackout Screen", "⌨ Key B", "KeyB"),
        "Pinch In / Out": ("Zoom Slide", "⊕ Ctrl + / -", "Ctrl + / -"),
        "Index Finger Point": ("Laser Pointer", "✦ Key L", "KeyL")
    }

    def __init__(self, root):
        self.root = root
        self.root.title("WAVE - Real-Time Gesture Recognition and Control Module")
        self.root.geometry("1280x830")
        self.root.minsize(1150, 780)
        self.root.configure(bg=self.BG_MAIN)

        self.ipc_config = IPCConfigManager(CONFIG_FILE)
        self.config = self.ipc_config.state
        self.os_dispatcher = OSActionDispatcher()
        self.vision_engine = VisionGestureEngine(MODEL_FILE)
        self.zoom_controller = TabletPinchZoomController()

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=max(0.7, float(self.config.get("min_detection_confidence", 0.7))),
            min_tracking_confidence=0.7
        )

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 60)

        self.frame_count = 0
        self.cached_landmarks = None
        self.fist_hold_start = 0.0
        self.FIST_HOLD_DURATION = 1.5
        self.last_fist_toggle_time = 0.0

        self.direction_banner_text = "DIRECTION: READY"
        self.fps_ema = ExponentialMovingAverage(alpha=0.15, fast_alpha=0.3)
        self.prev_frame_tick = cv2.getTickCount()
        self.rebinding_gesture = None

        self.assets = {
            "logo": GraphicsFactory.create_wave_logo(34),
            "avatar": GraphicsFactory.create_avatar_icon(32),
            "toggle_on": GraphicsFactory.create_toggle_switch(42, 24, True),
            "toggle_off": GraphicsFactory.create_toggle_switch(42, 24, False),
            "icons": {
                g: GraphicsFactory.create_gesture_icon(g, 44)
                for g in ["Right Swipe", "Left Swipe", "Open Palm", "Pinch In / Out", "Index Finger Point"]
            }
        }

        self._build_navbar()
        self._build_views()
        self._bind_shortcuts()

        self.switch_tab(self.config.get("active_tab", "live"))
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

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
            padx=14, pady=5, cursor="hand2"
        )
        self.tab_live_btn.pack(side="left")
        self.tab_live_btn.bind("<Button-1>", lambda e: self.switch_tab("live"))

        self.tab_map_btn = tk.Label(
            tab_outer, text="Keyboard Mapping", font=("Segoe UI", 9, "bold"),
            padx=14, pady=5, cursor="hand2"
        )
        self.tab_map_btn.pack(side="left")
        self.tab_map_btn.bind("<Button-1>", lambda e: self.switch_tab("mapping"))

        self.tab_calib_btn = tk.Label(
            tab_outer, text="Calibration", font=("Segoe UI", 9, "bold"),
            padx=14, pady=5, cursor="hand2"
            )
        self.tab_calib_btn.pack(side="left")
        self.tab_calib_btn.bind("<Button-1>", lambda e: self.switch_tab("calibration"))

        right_frame = tk.Frame(self.navbar, bg=self.BG_CARD)
        right_frame.pack(side="right", padx=24, pady=12)

        avatar_lbl = tk.Label(right_frame, image=self.assets["avatar"], bg=self.BG_CARD, cursor="hand2")
        avatar_lbl.pack(side="right", padx=(14, 0))

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
        sep.pack(side="right", padx=14)

        self.cam_pill = tk.Frame(right_frame, bg="#F1F5F9", highlightthickness=1, highlightbackground=self.BORDER_LIGHT, padx=10, pady=4)
        self.cam_pill.pack(side="right")

        self.cam_dot = tk.Label(self.cam_pill, text="●", font=("Segoe UI", 8), fg=self.GREEN_TEXT, bg="#F1F5F9")
        self.cam_dot.pack(side="left", padx=(0, 4))

        self.cam_txt = tk.Label(self.cam_pill, text="CAM ACTIVE", font=("Segoe UI", 8, "bold"), fg=self.SLATE_700, bg="#F1F5F9")
        self.cam_txt.pack(side="left")

    def _build_views(self):
        self.content_container = tk.Frame(self.root, bg=self.BG_MAIN)
        self.content_container.pack(side="top", fill="both", expand=True)

        self.live_view = tk.Frame(self.content_container, bg=self.BG_MAIN)
        self.mapping_view = tk.Frame(self.content_container, bg=self.BG_MAIN)
        self.calibration_view = tk.Frame(self.content_container, bg=self.BG_MAIN)

        self._build_live_presentation_view()
        self._build_keyboard_mapping_view()
        self._build_calibration_view()

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
            top_row, text="Right Swipe", font=("Segoe UI", 14, "bold"), fg=self.SLATE_900, bg=self.BG_CARD
        )
        self.live_gesture_title.pack(side="left", padx=(0, 10))

        self.recognized_badge = tk.Label(
            top_row, text=" RECOGNIZED ", font=("Segoe UI", 8, "bold"),
            fg=self.GREEN_TEXT, bg=self.GREEN_BG, padx=6, pady=2
        )
        self.recognized_badge.pack(side="left")

        bottom_row = tk.Frame(text_col, bg=self.BG_CARD)
        bottom_row.pack(side="top", anchor="w", pady=(4, 0))

        tk.Label(
            bottom_row, text="Dispatched: ", font=("Segoe UI", 9), fg=self.SLATE_500, bg=self.BG_CARD
        ).pack(side="left")

        self.dispatched_action_lbl = tk.Label(
            bottom_row, text="Next Slide", font=("Segoe UI", 9, "bold"), fg=self.BLUE_PRIMARY, bg=self.BG_CARD
        )
        self.dispatched_action_lbl.pack(side="left", padx=(0, 8))

        self.dispatched_key_badge = tk.Label(
            bottom_row, text="→ ArrowRight", font=("Consolas", 8, "bold"),
            fg=self.SLATE_700, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=6, pady=1
        )
        self.dispatched_key_badge.pack(side="left")

        pills_frame = tk.Frame(self.telemetry_card, bg=self.BG_CARD)
        pills_frame.pack(side="right")

        self.pill_cam_dot, self.pill_cam_val = self._create_status_pill(
            pills_frame, "CAMERA:", "CONNECTED", self.GREEN_TEXT
        )
        self.pill_hand_dot, self.pill_hand_val = self._create_status_pill(
            pills_frame, "HAND:", "DETECTED", self.BLUE_PRIMARY
        )
        self.pill_mode_dot, self.pill_mode_val = self._create_status_pill(
            pills_frame, "MODE:", "ON", self.GREEN_TEXT
        )

        self.footer_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=16, pady=10
        )
        self.footer_card.pack(side="top", fill="x")

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

        tk.Frame(self.footer_card, bg=self.BORDER_LIGHT, width=1, height=20).pack(side="left", padx=16)

        tk.Label(
            self.footer_card, text="SWIPE CONFIDENCE: ", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        ).pack(side="left")

        self.confidence_val_lbl = tk.Label(
            self.footer_card, text="99.4%", font=("Segoe UI", 9, "bold"),
            fg=self.GREEN_TEXT, bg=self.BG_CARD
        )
        self.confidence_val_lbl.pack(side="left")

        self.fps_footer_lbl = tk.Label(
            self.footer_card, text="FPS: 60.0   |   ZOOM: IDLE", font=("Segoe UI", 8, "bold"),
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

    def _build_keyboard_mapping_view(self):
        wrapper = tk.Frame(self.mapping_view, bg=self.BG_MAIN)
        wrapper.pack(fill="both", expand=True, padx=48, pady=24)

        header_row = tk.Frame(wrapper, bg=self.BG_MAIN)
        header_row.pack(fill="x", pady=(0, 18))

        title_col = tk.Frame(header_row, bg=self.BG_MAIN)
        title_col.pack(side="left", anchor="w")

        tk.Label(
            title_col, text="☷  HARDWARE ABSTRACTION LAYER", font=("Segoe UI", 8, "bold"),
            fg=self.BLUE_PRIMARY, bg=self.BG_MAIN
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            title_col, text="Gesture Key Mappings", font=("Segoe UI", 22, "bold"),
            fg=self.SLATE_900, bg=self.BG_MAIN
        ).pack(anchor="w", pady=(0, 4))

        tk.Label(
            title_col,
            text="Configure spatial hand tracking vectors to emulate native operating system keycodes\nand presentation software macros.",
            font=("Segoe UI", 10), fg=self.SLATE_500, bg=self.BG_MAIN, justify="left"
        ).pack(anchor="w")

        hid_card = tk.Frame(
            header_row, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=16, pady=10
        )
        hid_card.pack(side="right", anchor="s")

        tk.Label(hid_card, text="●", font=("Segoe UI", 10), fg=self.BLUE_PRIMARY, bg=self.BG_CARD).pack(side="left", padx=(0, 8))
        hid_text_col = tk.Frame(hid_card, bg=self.BG_CARD)
        hid_text_col.pack(side="left")
        tk.Label(hid_text_col, text="VIRTUAL HID STATUS", font=("Segoe UI", 7, "bold"), fg=self.SLATE_400, bg=self.BG_CARD).pack(anchor="w")
        tk.Label(hid_text_col, text="SYNCHRONIZED (0.4ms)", font=("Consolas", 9, "bold"), fg=self.BLUE_PRIMARY, bg=self.BG_CARD).pack(anchor="w")

        toolbar_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=18, pady=12
        )
        toolbar_card.pack(fill="x", pady=(0, 20))

        tk.Label(toolbar_card, text="⚙  ACTIVE PRESET", font=("Segoe UI", 8, "bold"), fg=self.SLATE_500, bg=self.BG_CARD).pack(side="left", padx=(0, 12))

        preset_box = tk.Frame(toolbar_card, bg=self.BG_MAIN, highlightthickness=1, highlightbackground=self.BORDER_LIGHT, padx=12, pady=6)
        preset_box.pack(side="left")
        tk.Label(preset_box, text="PowerPoint / Keynote Preset (Default)   ⌄", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_MAIN).pack(side="left")

        tk.Label(toolbar_card, text="  •   5 ACTIVE BINDINGS", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_CARD).pack(side="left", padx=10)

        new_map_btn = tk.Label(
            toolbar_card, text="+ New Mapping", font=("Segoe UI", 9, "bold"),
            fg="#FFFFFF", bg=self.BLUE_PRIMARY, padx=16, pady=7, cursor="hand2"
        )
        new_map_btn.pack(side="right", padx=(10, 0))
        new_map_btn.bind("<Button-1>", lambda e: self.reset_default_mappings())

        export_btn = tk.Label(
            toolbar_card, text="↓ EXPORT JSON", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_700, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=14, pady=7, cursor="hand2"
        )
        export_btn.pack(side="right")
        export_btn.bind("<Button-1>", lambda e: self.export_json_dialog())

        col_header = tk.Frame(wrapper, bg=self.BG_MAIN)
        col_header.pack(fill="x", padx=16, pady=(0, 8))
        tk.Label(col_header, text="ASSIGNED MOTION & KINEMATICS", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="left")
        tk.Label(col_header, text="STATE", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="right", padx=(0, 8))
        tk.Label(col_header, text="OUTPUT EMULATION", font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_MAIN).pack(side="right", padx=(0, 145))

        self.rows_container = tk.Frame(wrapper, bg=self.BG_MAIN)
        self.rows_container.pack(fill="both", expand=True)

        self.row_widgets = {}
        ordered_gestures = ["Right Swipe", "Left Swipe", "Open Palm", "Pinch In / Out", "Index Finger Point"]

        for gesture in ordered_gestures:
            row_card = tk.Frame(
                self.rows_container, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
                padx=16, pady=10
            )
            row_card.pack(fill="x", pady=(0, 10))

            tk.Label(row_card, image=self.assets["icons"][gesture], bg=self.BG_CARD).pack(side="left", padx=(0, 14))
            tk.Label(row_card, text=gesture, font=("Segoe UI", 11, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")

            is_enabled = self.config.get("gesture_states", {}).get(gesture, True)
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

            desc_text, default_pill, _ = self.DESCRIPTIONS.get(gesture, ("(Action)", "Key", "Key"))
            tk.Label(
                row_card, text=f"({desc_text})", font=("Segoe UI", 9),
                fg=self.SLATE_400, bg=self.BG_CARD, width=15, anchor="w"
            ).pack(side="right", padx=(8, 0))

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
        new_tab = "mapping" if self.config.get("active_tab", "live") == "live" else "live"
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
        elif event.keysym.lower() == 'q' or event.keysym == 'Escape':
            self.on_close()

    def switch_tab(self, tab_name):
        self.config["active_tab"] = tab_name
        self.ipc_config.save(self.config)

        self.live_view.pack_forget()
        self.mapping_view.pack_forget()
        if hasattr(self, 'calibration_view'):
            self.calibration_view.pack_forget()

        if tab_name == "live":
            self.live_view.pack(fill="both", expand=True)
            self.tab_live_btn.configure(bg=self.BG_CARD, fg=self.SLATE_900)
            self.tab_map_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
            self.tab_calib_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
        elif tab_name == "mapping":
            self.mapping_view.pack(fill="both", expand=True)
            self.tab_map_btn.configure(bg=self.BG_CARD, fg=self.BLUE_PRIMARY)
            self.tab_live_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
            self.tab_calib_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
        elif tab_name == "calibration":
            self.calibration_view.pack(fill="both", expand=True)
            self.tab_calib_btn.configure(bg=self.BG_CARD, fg=self.BLUE_PRIMARY)
            self.tab_live_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)
            self.tab_map_btn.configure(bg=self.PILL_BG, fg=self.SLATE_500)

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
            widgets["toggle"].configure(image=self.assets["toggle_on"])

    def _draw_pinch_hud(self, img, pt_thumb, pt_index, zoom_state, smoothed_ratio):
        mid_x = (pt_thumb[0] + pt_index[0]) // 2
        mid_y = (pt_thumb[1] + pt_index[1]) // 2

        if zoom_state == TabletPinchZoomController.STATE_ARMED:
            line_color = (0, 180, 255)
            cv2.line(img, pt_thumb, pt_index, line_color, 3, cv2.LINE_AA)
            cv2.circle(img, pt_thumb, 8, line_color, -1, cv2.LINE_AA)
            cv2.circle(img, pt_index, 8, line_color, -1, cv2.LINE_AA)
            cv2.circle(img, (mid_x, mid_y), 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(img, "LOCKED", (mid_x + 12, mid_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, line_color, 2, cv2.LINE_AA)

        elif zoom_state == TabletPinchZoomController.STATE_ZOOMING:
            line_color = (236, 109, 19)
            cv2.line(img, pt_thumb, pt_index, line_color, 4, cv2.LINE_AA)
            cv2.circle(img, pt_thumb, 10, line_color, -1, cv2.LINE_AA)
            cv2.circle(img, pt_index, 10, line_color, -1, cv2.LINE_AA)
            cv2.circle(img, (mid_x, mid_y), 8, (129, 185, 16), -1, cv2.LINE_AA)
            scale_pct = int(np.clip((smoothed_ratio / 0.34) * 100, 50, 350))
            cv2.putText(img, f"{scale_pct}%", (mid_x + 14, mid_y + 5),
                        cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

    def _update_telemetry_card(self, gesture_name, confidence, custom_action=None):
        icon_key = gesture_name if gesture_name in self.assets["icons"] else "Pinch In / Out"
        self.live_gesture_icon_lbl.configure(image=self.assets["icons"][icon_key])
        self.live_gesture_title.configure(text=gesture_name)
        self.recognized_badge.configure(text=" RECOGNIZED ", fg=self.GREEN_TEXT, bg=self.GREEN_BG)

        desc, _, key_tag = self.DESCRIPTIONS.get(gesture_name, (custom_action or "Zoom Slide", "", "Ctrl + / -"))
        self.dispatched_action_lbl.configure(text=custom_action if custom_action else desc)
        self.dispatched_key_badge.configure(text=key_tag)
        self.confidence_val_lbl.configure(text=f"{confidence:.1f}%")

    def update_loop(self):
        now_tick = cv2.getTickCount()
        dt_sec = max((now_tick - self.prev_frame_tick) / cv2.getTickFrequency(), 1e-4)
        self.prev_frame_tick = now_tick
        smooth_fps = self.fps_ema.filter(1.0 / dt_sec)

        self.config = self.ipc_config.poll_if_needed(interval=2.0)
        recognition_on = self.config.get("recognition_enabled", True)
        self._sync_engine_ui(recognition_on)

        ret, frame = self.cap.read()


        if not ret:
            self.pill_cam_dot.configure(fg="#EF4444")
            self.pill_cam_txt.configure(text=" DISCONNECTED")
            self._show_camera_error_popup()
            self.root.after(2000, self.update_loop)
            return

        if ret:
            frame = cv2.flip(frame, 1)
            lighting_sens = float(self.config.get("lighting_sensitivity", 0.6))
            if lighting_sens != 0.6: #it is only adjusted if user changed from default
                brightness_factor = 0.5 + (lighting_sens * 1.0)
                frame = cv2.convertScaleAbs(frame, alpha=brightness_factor, beta=0)
            # YOUR CODE: Apply ROI cropping
            if self.config.get("enable_roi_optional", False):
                roi = self.config.get("roi_settings", {})
                h, w = frame.shape[:2]
                x_min = int(roi.get("x_min", 0.15) * w)
                y_min = int(roi.get("y_min", 0.15) * h)
                x_max = int(roi.get("x_max", 0.85) * w)
                y_max = int(roi.get("y_max", 0.85) * h)
                frame = frame[y_min:y_max, x_min:x_max]
                # Draw ROI rectangle for visual feedback
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                self.frame_count += 1
            swipe_cooldown = float(self.config.get("swipe_cooldown", 0.6))

            frame_skip = max(2, int(self.config.get("frame_skip", 2)))
            active_zoom_lock = self.zoom_controller.state in (
                TabletPinchZoomController.STATE_ARMED,
                TabletPinchZoomController.STATE_ZOOMING
            )
            should_run_vision = active_zoom_lock or (self.frame_count % frame_skip == 0)

            if should_run_vision:
                small_frame = cv2.resize(frame, (640, 480))
                rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                rgb_small.flags.writeable = False
                results = self.hands.process(rgb_small)
                self.cached_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

            hand_detected = self.cached_landmarks is not None
            now_time = time.time()

            if hand_detected:
                self.pill_hand_dot.configure(fg=self.BLUE_PRIMARY)
                self.pill_hand_val.configure(text=" DETECTED")
            else:
                self.pill_hand_dot.configure(fg=self.SLATE_400)
                self.pill_hand_val.configure(text=" NONE")
                self.zoom_controller.reset()
                self.vision_engine.reset_tracking()
                self.fist_hold_start = 0.0

            card_w = max(self.video_card.winfo_width(), 860)
            card_h = max(self.video_card.winfo_height(), 460)
            display_frame = cv2.resize(frame, (card_w, card_h))

            if hand_detected:
                wrist = self.cached_landmarks.landmark[0]
                thumb_tip = self.cached_landmarks.landmark[4]
                index_tip = self.cached_landmarks.landmark[8]

                # 1. Closed Fist Hold Check (1.5s to Toggle Engine ON/OFF)
                is_fist = self.vision_engine.is_closed_fist(self.cached_landmarks)
                if is_fist and (now_time - self.last_fist_toggle_time) > 2.0:
                    self.zoom_controller.reset()
                    if self.fist_hold_start == 0.0:
                        self.fist_hold_start = now_time
                    elif (now_time - self.fist_hold_start) >= self.FIST_HOLD_DURATION:
                        self.toggle_engine()
                        recognition_on = self.config.get("recognition_enabled", True)
                        self.last_fist_toggle_time = now_time
                        self.fist_hold_start = 0.0
                else:
                    self.fist_hold_start = 0.0

                # Render Hand Skeleton Overlay
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

                # Evaluate static directional pose (Left / Right pointing vector)
                static_dir_gesture, dir_banner = self.vision_engine.detect_static_direction_pose(self.cached_landmarks)
                self.direction_banner_text = dir_banner

                if recognition_on:
                    if should_run_vision:
                        velocity_x, velocity_swipe = self.vision_engine.compute_velocity_and_swipe(wrist.x, now_time)
                    else:
                        velocity_x, velocity_swipe = 0.0, None

                    zoom_state = self.zoom_controller.state
                    zoom_ticks = 0
                    smoothed_ratio = self.zoom_controller.last_smoothed_ratio
                    zoom_label = ""

                    # Only evaluate pinch-to-zoom if NOT pointing strongly Left or Right
                    zoom_allowed = (
                        self.config.get("enable_zoom_optional", True)
                        and self.config.get("gesture_states", {}).get("Pinch In / Out", True)
                        and not is_fist
                        and static_dir_gesture is None
                    )
                    if zoom_allowed and should_run_vision:
                        zoom_state, zoom_ticks, smoothed_ratio, zoom_label = self.zoom_controller.update(
                            self.cached_landmarks, velocity_x
                        )
                    elif static_dir_gesture is not None:
                        self.zoom_controller.reset()
                        zoom_state = TabletPinchZoomController.STATE_IDLE

                    # Priority 1: Active Tablet Pinch-to-Zoom Mode
                    if zoom_state in (TabletPinchZoomController.STATE_ARMED, TabletPinchZoomController.STATE_ZOOMING):
                        self._draw_pinch_hud(display_frame, pt_thumb, pt_index, zoom_state, smoothed_ratio)
                        self.direction_banner_text = f"ZOOM: {zoom_label}"

                        if zoom_ticks != 0:
                            self.os_dispatcher.execute_zoom(zoom_ticks)
                            z_desc = "Zoom In (+)" if zoom_ticks > 0 else "Zoom Out (-)"
                            self._update_telemetry_card("Pinch In / Out", 99.4, custom_action=z_desc)
                            self.vision_engine.last_trigger_time = now_time

                        self.vision_engine.prediction_buffer.clear()

                    # Priority 2: Static Directional Pointing OR Dynamic Velocity Swipes
                    else:
                        cv2.arrowedLine(display_frame, pt_wrist, pt_index, (0, 0, 255), 4, tipLength=0.15, lineType=cv2.LINE_AA)

                        if should_run_vision:
                            mlp_label, mlp_conf = self.vision_engine.classify_static_mlp(self.cached_landmarks)

                            # 1st choice: Fast dynamic swipe
                            if velocity_swipe is not None:
                                detected_gesture = velocity_swipe
                                confidence = 0.994
                            # 2nd choice: Deliberate static directional point (Left/Right arrow vector)
                            elif static_dir_gesture is not None:
                                detected_gesture = static_dir_gesture
                                confidence = 0.985
                            # 3rd choice: Trained MLP static class
                            else:
                                detected_gesture = mlp_label
                                confidence = mlp_conf

                            self.vision_engine.prediction_buffer.append(detected_gesture)

                            # Require 2 consistent frames for static poses so it's deliberate & accurate
                            required_frames = 1 if velocity_swipe is not None else 2
                            if (
                                detected_gesture != "Neutral"
                                and (now_time - self.vision_engine.last_trigger_time) > swipe_cooldown
                            ):
                                if self.vision_engine.prediction_buffer.count(detected_gesture) >= required_frames:
                                    action_res = self.os_dispatcher.execute_shortcut(
                                        detected_gesture,
                                        self.config.get("gesture_mappings", {}),
                                        self.config.get("gesture_states", {})
                                    )
                                    if action_res != "Disabled":
                                        self._update_telemetry_card(detected_gesture, confidence * 100.0)
                                        self.vision_engine.last_trigger_time = now_time
                                        self.vision_engine.reset_tracking()

            # Draw Top-Left Green Overlay Text on Video Feed
            overlay_text = self.direction_banner_text if recognition_on else "ENGINE PAUSED"
            overlay_color = (50, 235, 50) if recognition_on else (60, 60, 235)
            cv2.putText(
                display_frame, overlay_text, (16, 36),
                cv2.FONT_HERSHEY_DUPLEX, 1.05, overlay_color, 2, cv2.LINE_AA
            )

            self.fps_footer_lbl.configure(
                text=f"FPS: {smooth_fps:.1f}   |   ZOOM: {self.zoom_controller.state}"
            )

            if self.config.get("active_tab", "live") == "live":
                rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_display)
                self.tk_video_img = ImageTk.PhotoImage(image=pil_img)
                self.video_label.configure(image=self.tk_video_img)

        self.root.after(15, self.update_loop)

    def on_close(self):
        try:
            if self.cap.isOpened():
                self.cap.release()
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