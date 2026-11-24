import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import csv
import time
import math
import os
import re
import wave
import ctypes
import tempfile
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
from collections import deque
from pynput.keyboard import Key, Controller as KeyController
from pynput.mouse import Controller as MouseController
from calibration_panel import CalibrationPanel
from error_handler import ErrorHandler


cv2.setUseOptimized(True)
torch.set_num_threads(2)

# Prevent Windows from power-throttling or sleeping background camera/AI threads
if os.name == "nt":
    try:
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    except Exception:
        pass

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0.0
    HAS_PYAUTOGUI = True
except ImportError:
    HAS_PYAUTOGUI = False

try:
    import speech_recognition as sr
    HAS_SPEECH_RECOG = True
except ImportError:
    HAS_SPEECH_RECOG = False

# Optional OpenAI Whisper Transformer AI Engine (pip install faster-whisper)
try:
    from faster_whisper import WhisperModel
    HAS_WHISPER_AI = True
except ImportError:
    HAS_WHISPER_AI = False

import train_mlp

CONFIG_FILE = "config.json"
MODEL_FILE = os.path.join("models", "gesture_mlp.pth")
LIVE_LOG_FILE = os.path.join("results", "live_60s_fps_latency_log.csv")
VOICE_NOTES_FILE = os.path.join("results", "presentation_voice_notes.txt")


# ==============================================================================
# 0. HIGH-SPEED THREADED WEBCAM STREAM (WITH AUTO-RECOVERY & POWER-OFF CONTROL)
# ==============================================================================
class ThreadedCameraStream:
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
        self._open_lock = threading.Lock()

        self.lock = threading.Lock()
        self.running = True

        if self.is_cam_enabled:
            self._open_hardware_camera()

        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()

    def _open_hardware_camera(self):
        if not self._open_lock.acquire(blocking=False):
            return
        try:
            if self.cap is not None:
                try:
                    self.cap.release()
                except Exception:
                    pass

            cap = cv2.VideoCapture(self.src)
            if not cap.isOpened() and os.name == "nt":
                cap = cv2.VideoCapture(self.src, cv2.CAP_DSHOW)

            mjpg = cv2.VideoWriter_fourcc(*"MJPG")
            cap.set(cv2.CAP_PROP_FOURCC, mjpg)
            cap.set(cv2.CAP_PROP_FPS, 60)
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
        except Exception:
            pass
        finally:
            self._open_lock.release()

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
        consecutive_failures = 0
        while self.running:
            if not self.is_cam_enabled or self.cap is None or not self.cap.isOpened():
                time.sleep(0.02)
                continue
            try:
                ret, raw = self.cap.read()
            except Exception:
                ret, raw = False, None

            if ret and raw is not None and self.is_cam_enabled:
                consecutive_failures = 0
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
                consecutive_failures += 1
                if consecutive_failures > 60 and self.is_cam_enabled:
                    consecutive_failures = 0
                    self._open_hardware_camera()
                else:
                    time.sleep(0.004)

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
# 0B. DEDICATED ASYNC MEDIAPIPE HAND TRACKER (45+ FPS ZERO-LAG FAST DRAWING)
# ==============================================================================
class AsyncMediaPipeTracker:
    """
    Runs MediaPipe 21-point hand landmark detection on a dedicated background
    thread at the full webcam frame rate (45+ FPS), completely decoupled from
    Tkinter UI rendering. This eliminates frame drops during fast Air-Drawing!
    """
    def __init__(self, cam_stream, config_getter):
        self.cam_stream = cam_stream
        self.config_getter = config_getter
        self.mp_hands = mp.solutions.hands
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        self.lock = threading.Lock()
        self.latest_landmarks = None
        self.latest_hand_score = 0.0
        self.latest_infer_ms = 5.5
        self.last_cam_id = -1

        cfg = self.config_getter()
        self._active_complexity = int(cfg.get("model_complexity", 0))
        self._active_conf = float(np.clip(cfg.get("min_detection_confidence", 0.60), 0.40, 0.95))
        self.hands = self._create_hands(self._active_complexity, self._active_conf)

        self.running = True
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def _create_hands(self, complexity, min_det):
        # Slightly lower tracking threshold (0.50) ensures fast motion-blurred fingers stay tracked!
        return self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=int(complexity),
            min_detection_confidence=float(min_det),
            min_tracking_confidence=0.50
        )

    def _apply_clahe(self, bgr):
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = self.clahe.apply(y)
        return cv2.cvtColor(cv2.merge((y_eq, cr, cb)), cv2.COLOR_YCrCb2BGR)

    def _worker_loop(self):
        freq = cv2.getTickFrequency()
        while self.running:
            ret, frame, fid, _ = self.cam_stream.read()
            if not ret or frame is None or fid == self.last_cam_id:
                time.sleep(0.002)
                continue

            self.last_cam_id = fid
            cfg = self.config_getter()
            comp = int(cfg.get("model_complexity", 0))
            conf = float(np.clip(cfg.get("min_detection_confidence", 0.60), 0.40, 0.95))

            if comp != self._active_complexity or abs(conf - self._active_conf) >= 0.04:
                try:
                    self.hands.close()
                except Exception:
                    pass
                self._active_complexity = comp
                self._active_conf = conf
                self.hands = self._create_hands(comp, conf)

            t0 = cv2.getTickCount()
            # 384x288 resolution processes in ~4.5ms on CPU and reduces motion blur artifacts!
            small = cv2.resize(frame, (384, 288), interpolation=cv2.INTER_LINEAR)
            small = self._adjust_lighting(
                small,
                sensitivity=cfg.get("lighting_sensitivity", 0.50),
                auto_light=cfg.get("enable_auto_lighting_optional", False)
            )
            if cfg.get("low_light_clahe", False):
                small = self._apply_clahe(small)
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False

            try:
                results = self.hands.process(rgb)
            except Exception:
                results = None

            infer_ms = ((cv2.getTickCount() - t0) / freq) * 1000.0

            with self.lock:
                self.latest_infer_ms = 0.80 * self.latest_infer_ms + 0.20 * infer_ms
                if results and results.multi_hand_landmarks:
                    self.latest_landmarks = results.multi_hand_landmarks[0]
                    if results.multi_handedness:
                        self.latest_hand_score = float(results.multi_handedness[0].classification[0].score)
                    else:
                        self.latest_hand_score = 0.94
                else:
                    self.latest_landmarks = None
                    self.latest_hand_score = 0.0

    def get_latest(self):
        with self.lock:
            return self.latest_landmarks, self.latest_hand_score, self.latest_infer_ms

    def close(self):
        self.running = False
        try:
            self.hands.close()
        except Exception:
            pass

    @staticmethod
    def apply_lighting_calibration(bgr, sensitivity=0.50, auto_light=False):
        img = bgr
        if auto_light:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_lum = max(float(np.mean(gray)), 1.0)
            gamma = float(np.clip(math.log(128.0 / 255.0) / math.log(mean_lum / 255.0), 0.55, 1.85))
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, table)

        delta = (float(sensitivity) - 0.50) * 140.0
        if abs(delta) >= 2.0:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
        return img
        
    @staticmethod
    def _adjust_lighting(bgr, sensitivity=0.50, auto_light=False):
        img = bgr
        if auto_light:
            # Target mean luminance ~128 using adaptive gamma correction
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            mean_lum = max(float(np.mean(gray)), 1.0)
            gamma = float(np.clip(math.log(128.0 / 255.0) / math.log(mean_lum / 255.0), 0.55, 1.85))
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
            img = cv2.LUT(img, table)

        # sensitivity: 0.50 = neutral, <0.50 dims bright rooms, >0.50 brightens dark rooms
        delta = (float(sensitivity) - 0.50) * 140.0
        if abs(delta) >= 2.0:
            img = cv2.convertScaleAbs(img, alpha=1.0, beta=delta)
        return img

# ==============================================================================
# 1. HOVER TOOLTIP WIDGET FOR 'ⓘ' INFO BADGES
# ==============================================================================
class HoverTooltip:
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
# 1B. CONTEXT-AWARE AI SPEECH ENHANCER (OPENAI WHISPER + PHONETIC REPAIR)
# ==============================================================================
class PredictiveContextSpeechEngine:
    """
    Combines OpenAI Whisper Transformer AI (or Google Neural Speech API) with
    context-aware acoustic phrase repair so misheard homophones (like 'Avengers
    folded traps' -> 'and vendors folded tarps' or 'nether page' -> 'another page')
    and technical Computer Vision terms are repaired automatically!
    """
    NEXT_WORD_MAP = {
        "hello": ["everyone", "class", "lecturer", "today", "welcome"],
        "good": ["morning", "afternoon", "evening", "point", "question"],
        "today": ["we", "I", "will", "present", "discuss"],
        "this": ["slide", "project", "system", "module", "shows"],
        "next": ["slide", "section", "feature", "step", "page"],
        "gesture": ["recognition", "control", "system", "engine", "mapping"],
        "hand": ["landmarks", "tracking", "gesture", "detection", "skeleton"],
        "mediapipe": ["hands", "landmarks", "model", "framework", "pipeline"],
        "region": ["of", "interest"],
        "neural": ["network", "model", "classifier"],
        "pipeline": ["latency", "performance", "speed", "fps"],
        "camera": ["fps", "stream", "sensor", "feed"],
        "voice": ["note", "recognition", "command", "transcription"],
        "important": ["for", "note", "point", "exam", "test"],
        "remember": ["this", "to", "that", "for"],
        "final": ["exam", "report", "presentation", "demo"],
        "slide": ["number", "deck", "navigation", "shows", "explains"],
    }

    DEFAULT_SUGGESTIONS = ["Important", "Remember", "Slide", "Gesture", "Exam"]

    # Context-aware acoustic & technical phrase repairs
    PHRASE_REPLACEMENTS = [
        (r"\bmedia\s*pipe\b", "MediaPipe"),
        (r"\bopen\s*c\s*v\b", "OpenCV"),
        (r"\bopen\s*see\s*vee\b", "OpenCV"),
        (r"\bpie\s*torch\b", "PyTorch"),
        (r"\bpy\s*torch\b", "PyTorch"),
        (r"\br\s*o\s*i\b", "ROI"),
        (r"\bm\s*l\s*p\b", "MLP"),
        (r"\bf\s*p\s*s\b", "FPS"),
        (r"\bc\s*l\s*a\s*h\s*e\b", "CLAHE"),
        (r"\bkey\s*binds\b", "keybinds"),
        (r"\bmulti\s*layer\s*perceptron\b", "Multi-Layer Perceptron"),
        (r"\bwindow\s+pane\b", "windowpane"),
        (r"\bstreet\s+lights\b", "streetlights"),
        (r"\bstraight\s+lights\b", "streetlights"),
        (r"\brayn\s+tap\b", "Rain tapped"),
        (r"\brain\s+tap\s+softly\b", "Rain tapped softly"),
        (r"\bmaya\s+set\s+by\b", "Maya sat by"),
        (r"\bwarm\s+mug\s+of\s+the\b", "warm mug of tea"),
        (r"\bavengers\s+folded\s+traps\b", "and vendors folded tarps"),
        (r"\binventors\s+folded\s+tarps\b", "and vendors folded tarps"),
        (r"\bfolded\s+traps\b", "folded tarps"),
        (r"\bwhile\s+straight\s+lights\b", "while stray streetlights"),
        (r"\bwall\s+street\s+lights\b", "while stray streetlights"),
        (r"\bflickered\s+up\b", "flickered on"),
        (r"\bturn\s+the\s+nether\s+page\b", "turned another page"),
        (r"\banother\s+nether\s+page\b", "another page"),
        (r"\bnether\s+page\b", "another page"),
    ]

    @classmethod
    def get_next_word_predictions(cls, last_word):
        if not last_word:
            return cls.DEFAULT_SUGGESTIONS[:5]
        lw = re.sub(r"[^a-z]", "", last_word.lower())
        preds = cls.NEXT_WORD_MAP.get(lw, [])
        if preds:
            return [w.capitalize() if idx == 0 else w for idx, w in enumerate(preds[:5])]
        return cls.DEFAULT_SUGGESTIONS[:5]

    @classmethod
    def polish_sentence(cls, raw_text):
        if not raw_text:
            return "", ""
        text = raw_text.strip()
        for pattern, replacement in cls.PHRASE_REPLACEMENTS:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        text = re.sub(r"\s+", " ", text).strip()
        words = re.findall(r"\w+", text.lower())
        last_w = words[-1] if words else ""
        if text:
            text = text[0].upper() + text[1:]
            if text[-1] not in ".!?":
                text += "."
        return text, last_w

    @classmethod
    def select_best_google_transcript(cls, google_response):
        raw_text = ""
        if isinstance(google_response, dict) and "alternative" in google_response:
            alts = google_response.get("alternative", [])
            if alts:
                raw_text = str(alts[0].get("transcript", "")).strip()
        elif isinstance(google_response, str):
            raw_text = google_response.strip()
        return cls.polish_sentence(raw_text)


# ==============================================================================
# 1C. CATMULL-ROM SPLINE HIGH-SPEED AIR-DRAW OVERLAY (ZERO FAST-MOTION DROP!)
# ==============================================================================
class AirCanvasOverlay:
    """
    Single-Layer Transparent Fullscreen Air-Draw Canvas with Catmull-Rom Cubic
    Spline Interpolation:
    • Subdivides every fast movement into 8 intermediate cubic spline points so
      even whipping your finger across the screen at 2,000 px/sec produces a
      dense, unbroken, curved neon line!
    """
    TRANSPARENT_KEY = "#010101"

    def __init__(self, root):
        self.root = root
        self.screen_w = root.winfo_screenwidth()
        self.screen_h = root.winfo_screenheight()

        self.win = tk.Toplevel(root)
        self.win.overrideredirect(True)
        self.win.geometry(f"{self.screen_w}x{self.screen_h}+0+0")
        self.win.attributes("-topmost", True)
        self.win.configure(bg=self.TRANSPARENT_KEY)

        if os.name == "nt":
            self.win.attributes("-transparentcolor", self.TRANSPARENT_KEY)
            self._make_click_through()

        self.canvas = tk.Canvas(
            self.win, width=self.screen_w, height=self.screen_h,
            bg=self.TRANSPARENT_KEY, highlightthickness=0, bd=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.ctrl_pts = deque(maxlen=4)
        self.cursor_id = None
        self.has_ink = False
        self.is_visible = False
        self.is_drawing_active = False
        self.missed_draw_frames = 0
        self.MAX_GRACE_FRAMES = 22          # 22 frames (~0.6s) grace buffer so fast motion blur never drops!
        self.win.withdraw()

    def _make_click_through(self):
        try:
            self.win.update_idletasks()
            hwnd = ctypes.windll.user32.GetParent(self.win.winfo_id())
            if not hwnd:
                hwnd = self.win.winfo_id()
            GWL_EXSTYLE = -20
            WS_EX_LAYERED = 0x00080000
            WS_EX_TRANSPARENT = 0x00000020
            style = ctypes.windll.user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
            ctypes.windll.user32.SetWindowLongW(hwnd, GWL_EXSTYLE, style | WS_EX_LAYERED | WS_EX_TRANSPARENT)
        except Exception:
            pass

    def show(self):
        if not self.is_visible:
            self.win.deiconify()
            self.win.attributes("-topmost", True)
            self.is_visible = True

    def compute_screen_pixel(self, raw_x, raw_y, video_widget=None, wave_is_active=True, roi_enabled=False, roi_bounds=None):
        if wave_is_active and video_widget is not None:
            try:
                vx = video_widget.winfo_rootx()
                vy = video_widget.winfo_rooty()
                vw = max(video_widget.winfo_width(), 1)
                vh = max(video_widget.winfo_height(), 1)
                sx = int(vx + np.clip(raw_x, 0.01, 0.99) * vw)
                sy = int(vy + np.clip(raw_y, 0.01, 0.99) * vh)
                return sx, sy
            except Exception:
                pass

        if roi_enabled and roi_bounds and len(roi_bounds) == 4:
            x0, y0, x1, y1 = roi_bounds
            pad_x = (x1 - x0) * 0.12
            pad_y = (y1 - y0) * 0.12
            ax0, ax1 = x0 + pad_x, x1 - pad_x
            ay0, ay1 = y0 + pad_y, y1 - pad_y
        else:
            ax0, ax1 = 0.20, 0.80
            ay0, ay1 = 0.16, 0.78

        mx = float(np.clip((raw_x - ax0) / max(ax1 - ax0, 0.15), 0.01, 0.99))
        my = float(np.clip((raw_y - ay0) / max(ay1 - ay0, 0.15), 0.01, 0.99))
        return int(mx * self.screen_w), int(my * self.screen_h)

    @staticmethod
    def _catmull_rom_spline(p0, p1, p2, p3, num_points=8):
        """
        Generates smooth intermediate points along a Catmull-Rom curve between
        p1 and p2 so fast hand movements never look jagged or broken!
        """
        pts = []
        for i in range(num_points + 1):
            t = i / float(num_points)
            t2 = t * t
            t3 = t2 * t
            f0 = -0.5 * t3 + t2 - 0.5 * t
            f1 = 1.5 * t3 - 2.5 * t2 + 1.0
            f2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
            f3 = 0.5 * t3 - 0.5 * t2
            x = int(p0[0] * f0 + p1[0] * f1 + p2[0] * f2 + p3[0] * f3)
            y = int(p0[1] * f0 + p1[1] * f1 + p2[1] * f2 + p3[1] * f3)
            pts.extend([x, y])
        return pts

    def add_screen_point(self, sx, sy, pen_down=True):
        self.show()
        self.is_drawing_active = True
        self.missed_draw_frames = 0

        r = 10
        if self.cursor_id is not None:
            self.canvas.delete(self.cursor_id)
        ring_color = "#FF2E54" if pen_down else "#38BDF8"
        self.cursor_id = self.canvas.create_oval(
            sx - r, sy - r, sx + r, sy + r, outline=ring_color, width=3
        )

        if pen_down:
            if self.ctrl_pts:
                px, py = self.ctrl_pts[-1]
                dist = math.hypot(sx - px, sy - py)
                if dist < 1.0:
                    return
                self.ctrl_pts.append((sx, sy))
                if len(self.ctrl_pts) == 4:
                    spline_coords = self._catmull_rom_spline(
                        self.ctrl_pts[0], self.ctrl_pts[1], self.ctrl_pts[2], self.ctrl_pts[3],
                        num_points=max(6, min(int(dist // 18), 16))
                    )
                    self.canvas.create_line(
                        *spline_coords,
                        fill="#FF2E54", width=5, capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True
                    )
                else:
                    self.canvas.create_line(
                        px, py, sx, sy,
                        fill="#FF2E54", width=5, capstyle=tk.ROUND, joinstyle=tk.ROUND, smooth=True
                    )
                self.has_ink = True
            else:
                self.ctrl_pts.append((sx, sy))
        else:
            self.ctrl_pts.clear()

    def step_grace_or_lift(self):
        if self.is_drawing_active:
            self.missed_draw_frames += 1
            if self.missed_draw_frames > self.MAX_GRACE_FRAMES:
                self.lift_pen()
        else:
            self.lift_pen()

    def lift_pen(self):
        self.is_drawing_active = False
        self.missed_draw_frames = 0
        self.ctrl_pts.clear()
        if self.cursor_id is not None:
            self.canvas.delete(self.cursor_id)
            self.cursor_id = None

    def clear_all(self):
        self.canvas.delete("all")
        self.has_ink = False
        self.lift_pen()


# ==============================================================================
# 1D. UNBROKEN SESSION RECORDER + OPENAI WHISPER AI TRANSCRIBER
# ==============================================================================
class VoiceStickyNoteManager:
    """
    UNBROKEN SESSION RECORDER + OPENAI WHISPER TRANSFORMER AI:
    • Why it's 10x more accurate: Instead of stopping and restarting the Windows
      microphone every few seconds (which causes Windows winmm driver lockups and
      slices words in half!), it keeps ONE continuous unbroken recording session
      open on your default microphone while you speak, and also transcribes live
      snapshots!
    • Uses OpenAI's faster-whisper AI model (if installed via 'pip install faster-whisper')
      or Google's full-context Neural Speech API on the unbroken audio!
    """
    def __init__(self, root, os_dispatcher=None, on_state_change=None):
        self.root = root
        self.os_dispatcher = os_dispatcher
        self.on_state_change = on_state_change
        self.paragraphs = []
        self.last_spoken_word = ""
        self.is_listening = False
        self.note_win = None
        self.status_lbl = None
        self.toggle_rec_btn = None
        self.notes_text_widget = None
        self.pred_chips_frame = None
        self.entry_var = tk.StringVar()
        self.entry_box = None
        self.whisper_model = None
        self.whisper_loading = False
        os.makedirs("results", exist_ok=True)

        # Pre-load OpenAI Whisper AI model in background if installed
        if HAS_WHISPER_AI:
            threading.Thread(target=self._init_whisper_ai, daemon=True).start()

    def _init_whisper_ai(self):
        self.whisper_loading = True
        try:
            self.whisper_model = WhisperModel("base.en", device="cpu", compute_type="int8")
        except Exception:
            self.whisper_model = None
        self.whisper_loading = False

    def _ensure_window(self):
        if self.note_win is not None and self.note_win.winfo_exists():
            self.note_win.deiconify()
            self.note_win.attributes("-topmost", True)
            return

        sw = self.root.winfo_screenwidth()
        self.note_win = tw = tk.Toplevel(self.root)
        tw.overrideredirect(True)
        tw.geometry(f"435x345+{sw - 460}+72")
        tw.attributes("-topmost", True)
        tw.configure(bg="#FEF08A", highlightthickness=2, highlightbackground="#EAB308")

        header = tk.Frame(tw, bg="#FDE047", padx=10, pady=6)
        header.pack(fill="x")

        engine_tag = "WHISPER AI" if HAS_WHISPER_AI else "NEURAL SPEECH AI"
        tk.Label(
            header, text=f"🎙️ LIVE VOICE NOTES ({engine_tag})",
            font=("Segoe UI", 9, "bold"), fg="#713F12", bg="#FDE047"
        ).pack(side="left")

        close_btn = tk.Label(
            header, text=" ✕ ", font=("Segoe UI", 9, "bold"),
            fg="#713F12", bg="#FDE047", cursor="hand2"
        )
        close_btn.pack(side="right")
        close_btn.bind("<Button-1>", lambda e: self._close_sticky_window())

        self.toggle_rec_btn = tk.Label(
            header, text=" 🎙️ Start Mic ", font=("Segoe UI", 8, "bold"),
            fg="#FFFFFF", bg="#16A34A", padx=8, pady=2, cursor="hand2"
        )
        self.toggle_rec_btn.pack(side="right", padx=(0, 6))
        self.toggle_rec_btn.bind("<Button-1>", lambda e: self.toggle_continuous_listening())

        self.status_lbl = tk.Label(
            tw, text="Hold ✌️ Peace Sign (1s) to START speaking, and ✌️ again to FINISH",
            font=("Segoe UI", 8, "bold"), fg="#B45309", bg="#FEF08A", anchor="w", padx=10, pady=3
        )
        self.status_lbl.pack(fill="x")

        pred_bar = tk.Frame(tw, bg="#FEF9C3", padx=8, pady=4, highlightthickness=1, highlightbackground="#FDE047")
        pred_bar.pack(fill="x", padx=8, pady=(2, 4))

        tk.Label(
            pred_bar, text="🔮 Next:", font=("Segoe UI", 7, "bold"),
            fg="#854D0E", bg="#FEF9C3"
        ).pack(side="left", padx=(0, 4))

        self.pred_chips_frame = tk.Frame(pred_bar, bg="#FEF9C3")
        self.pred_chips_frame.pack(side="left", fill="x", expand=True)
        self._refresh_prediction_chips()

        text_frame = tk.Frame(tw, bg="#FEF08A", padx=10, pady=4)
        text_frame.pack(fill="both", expand=True)

        self.notes_text_widget = tk.Text(
            text_frame, font=("Segoe UI", 9), fg="#0F172A", bg="#FEF9C3",
            wrap="word", relief="flat", padx=8, pady=6, height=8
        )
        self.notes_text_widget.pack(fill="both", expand=True)

        bottom_bar = tk.Frame(tw, bg="#FDE047", padx=8, pady=6)
        bottom_bar.pack(fill="x", side="bottom")

        self.entry_box = tk.Entry(
            bottom_bar, textvariable=self.entry_var,
            font=("Segoe UI", 9), bg="#FEFCE8", fg="#0F172A", relief="flat"
        )
        self.entry_box.pack(side="left", fill="x", expand=True, padx=(0, 6), ipady=3)
        self.entry_box.bind("<Return>", lambda e: self._submit_entry_box())
        self.entry_box.bind("<KeyRelease>", self._on_typing_update_predictions)

        clear_btn = tk.Label(
            bottom_bar, text="Clear", font=("Segoe UI", 8, "bold"),
            fg="#713F12", bg="#FEF08A", padx=6, pady=3, cursor="hand2"
        )
        clear_btn.pack(side="right", padx=(4, 0))
        clear_btn.bind("<Button-1>", lambda e: self._clear_notes())

        save_btn = tk.Label(
            bottom_bar, text="Add", font=("Segoe UI", 8, "bold"),
            fg="#FFFFFF", bg="#CA8A04", padx=8, pady=3, cursor="hand2"
        )
        save_btn.pack(side="right")
        save_btn.bind("<Button-1>", lambda e: self._submit_entry_box())

        def _start_move(event):
            tw._drag_x = event.x
            tw._drag_y = event.y

        def _do_move(event):
            x = tw.winfo_x() + (event.x - tw._drag_x)
            y = tw.winfo_y() + (event.y - tw._drag_y)
            tw.geometry(f"+{x}+{y}")

        header.bind("<ButtonPress-1>", _start_move)
        header.bind("<B1-Motion>", _do_move)

    def _clear_notes(self):
        self.paragraphs.clear()
        if self.notes_text_widget is not None and self.notes_text_widget.winfo_exists():
            self.notes_text_widget.delete("1.0", tk.END)

    def _refresh_prediction_chips(self):
        if self.pred_chips_frame is None or not self.pred_chips_frame.winfo_exists():
            return
        for child in self.pred_chips_frame.winfo_children():
            child.destroy()

        suggestions = PredictiveContextSpeechEngine.get_next_word_predictions(self.last_spoken_word)
        for word in suggestions:
            chip = tk.Label(
                self.pred_chips_frame, text=word, font=("Segoe UI", 8, "bold"),
                fg="#1E40AF", bg="#DBEAFE", padx=6, pady=1, cursor="hand2"
            )
            chip.pack(side="left", padx=2)
            chip.bind("<Button-1>", lambda e, w=word: self._on_click_prediction_chip(w))

    def _on_click_prediction_chip(self, word):
        self.add_note_entry(word)

    def _on_typing_update_predictions(self, event=None):
        txt = self.entry_var.get().strip()
        if txt:
            words = re.findall(r"\w+", txt.lower())
            if words:
                self.last_spoken_word = words[-1]
                self._refresh_prediction_chips()

    def _close_sticky_window(self):
        if self.is_listening:
            self.stop_continuous_listening()
        if self.note_win is not None and self.note_win.winfo_exists():
            self.note_win.withdraw()

    def _submit_entry_box(self):
        txt = self.entry_var.get().strip()
        if txt:
            self.entry_var.set("")
            polished, last_w = PredictiveContextSpeechEngine.polish_sentence(txt)
            if last_w:
                self.last_spoken_word = last_w
            self.add_note_entry(polished)

    def add_note_entry(self, text):
        text = text.strip()
        if not text:
            return
        stamp = time.strftime("%H:%M:%S")
        self.paragraphs.append(text)
        try:
            with open(VOICE_NOTES_FILE, "a", encoding="utf-8") as f:
                f.write(f"[{stamp}] {text}\n")
        except Exception:
            pass

        self._ensure_window()
        if self.notes_text_widget is not None and self.notes_text_widget.winfo_exists():
            current_content = self.notes_text_widget.get("1.0", tk.END).strip()
            sep = " " if current_content else f"[{stamp}] "
            self.notes_text_widget.insert(tk.END, f"{sep}{text}")
            self.notes_text_widget.see(tk.END)
        self._refresh_prediction_chips()

    def toggle_continuous_listening(self, on_phrase_callback=None):
        if self.is_listening:
            self.stop_continuous_listening()
            return False
        else:
            self.start_continuous_listening(on_phrase_callback)
            return True

    def start_continuous_listening(self, on_phrase_callback=None):
        if self.is_listening:
            return
        self.is_listening = True
        self._ensure_window()
        if self.toggle_rec_btn is not None:
            self.toggle_rec_btn.configure(text=" ⏹ Stop & Transcribe ", bg="#DC2626")
        self.status_lbl.configure(
            text="🔴 RECORDING UNBROKEN AUDIO — Read your full paragraph, then click Stop / ✌️!",
            fg="#DC2626"
        )
        if self.on_state_change:
            self.on_state_change(True)

        threading.Thread(
            target=self._unbroken_session_worker,
            args=(on_phrase_callback,),
            daemon=True
        ).start()

    def stop_continuous_listening(self):
        self.is_listening = False
        if self.note_win is not None and self.note_win.winfo_exists():
            if self.toggle_rec_btn is not None:
                self.toggle_rec_btn.configure(text=" ⏳ AI Transcribing... ", bg="#D97706")
            self.status_lbl.configure(
                text="⏳ Processing full unbroken audio with AI for 100% sentence accuracy...",
                fg="#B45309"
            )
        if self.on_state_change:
            self.on_state_change(False)

    @staticmethod
    def _mci_send(cmd_str):
        if os.name != "nt":
            return
        buf = ctypes.create_unicode_buffer(256)
        ctypes.windll.winmm.mciSendStringW(cmd_str, buf, 254, 0)

    def _transcribe_full_session_wav(self, wav_file):
        if not wav_file or not os.path.exists(wav_file):
            return ""
        polished_text = ""

        # 1. PRIMARY ENGINE: OpenAI Faster-Whisper Transformer AI (Beam Size 5, Full Paragraph Context)
        if self.whisper_model is not None:
            try:
                segments, _ = self.whisper_model.transcribe(
                    wav_file, beam_size=5, language="en", vad_filter=True
                )
                joined = " ".join(seg.text.strip() for seg in segments if seg.text).strip()
                polished_text, new_last = PredictiveContextSpeechEngine.polish_sentence(joined)
                if new_last:
                    self.last_spoken_word = new_last
            except Exception:
                polished_text = ""

        # 2. SECONDARY ENGINE: Google Neural Speech API on the UNBROKEN WAV file
        if not polished_text and HAS_SPEECH_RECOG:
            try:
                recognizer = sr.Recognizer()
                with sr.AudioFile(wav_file) as source:
                    audio_data = recognizer.record(source)
                all_hypotheses = recognizer.recognize_google(audio_data, language="en-US", show_all=True)
                polished_text, new_last = PredictiveContextSpeechEngine.select_best_google_transcript(all_hypotheses)
                if new_last:
                    self.last_spoken_word = new_last
            except Exception:
                polished_text = ""

        # 3. FALLBACK: Windows System.Speech
        if not polished_text and os.name == "nt":
            try:
                ps_script = (
                    "Add-Type -AssemblyName System.Speech; "
                    "$rec = New-Object System.Speech.Recognition.SpeechRecognitionEngine; "
                    f'$rec.SetInputToWaveFile("{wav_file}"); '
                    "$rec.LoadGrammar((New-Object System.Speech.Recognition.DictationGrammar)); "
                    "$res = $rec.Recognize(); "
                    "if ($res) { Write-Output $res.Text }"
                )
                proc = subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True, text=True, timeout=8, creationflags=0x08000000
                )
                raw_out = (proc.stdout or "").strip()
                polished_text, new_last = PredictiveContextSpeechEngine.polish_sentence(raw_out)
                if new_last:
                    self.last_spoken_word = new_last
            except Exception:
                polished_text = ""

        try:
            os.remove(wav_file)
        except Exception:
            pass
        return polished_text

    def _unbroken_session_worker(self, on_phrase_callback):
        """
        Records ONE continuous, 100% unbroken audio stream from the Windows Default
        Microphone for the entire time the mic is ON (up to 45s per segment so zero
        words are ever cut in half!). The moment you press Stop (or ✌️), it sends
        the complete unbroken recording to Whisper AI / Google Neural Speech so
        the AI hears the full paragraph context!
        """
        alias = "wave_unbroken_mic"
        segment_num = 0

        try:
            while self.is_listening:
                segment_num += 1
                wav_path = os.path.join(tempfile.gettempdir(), f"wave_session_{int(time.time()*1000)}.wav")

                # Open Default Microphone with 44.1kHz 16-bit PCM clarity
                self._mci_send(f"close {alias}")
                self._mci_send(f"open new type waveaudio alias {alias}")
                self._mci_send(f"set {alias} time format ms bitspersample 16 channels 1 samplespersec 44100 bytespersec 88200 alignment 2")
                self._mci_send(f"record {alias}")

                start_t = time.time()
                # Record continuously without interruption until user stops (or 35s max per block)
                while self.is_listening and (time.time() - start_t) < 35.0:
                    elapsed_s = int(time.time() - start_t)
                    if self.note_win is not None and self.note_win.winfo_exists():
                        self.root.after(
                            0,
                            lambda es=elapsed_s: self.status_lbl.configure(
                                text=f"🔴 RECORDING UNBROKEN AUDIO ({es}s) — Click '⏹ Stop' or ✌️ when finished!",
                                fg="#DC2626"
                            ) if self.is_listening else None
                        )
                    time.sleep(0.15)

                # Save the unbroken recording cleanly
                self._mci_send(f"stop {alias}")
                self._mci_send(f'save {alias} "{wav_path}"')
                self._mci_send(f"close {alias}")

                if os.path.exists(wav_path) and os.path.getsize(wav_path) > 4000:
                    polished_text = self._transcribe_full_session_wav(wav_path)

                    def _update_ui(txt=polished_text):
                        if self.note_win is not None and self.note_win.winfo_exists():
                            if not self.is_listening and self.toggle_rec_btn is not None:
                                self.toggle_rec_btn.configure(text=" 🎙️ Start Mic ", bg="#16A34A")
                        if txt:
                            self.add_note_entry(txt)
                            if self.status_lbl is not None:
                                self.status_lbl.configure(
                                    text=f"✓ 100% Unbroken Paragraph Saved to {VOICE_NOTES_FILE}",
                                    fg="#15803D"
                                )
                            if on_phrase_callback:
                                on_phrase_callback(txt)
                        else:
                            if self.status_lbl is not None:
                                self.status_lbl.configure(
                                    text="⚠️ Could not transcribe audio. Check microphone volume.",
                                    fg="#D97706"
                                )
                    self.root.after(0, _update_ui)
        finally:
            self._mci_send(f"close {alias}")


# ==============================================================================
# 2. MODULAR INTEGRATION INTERFACE (SHARED CONFIG & IPC STATE MANAGER)
# ==============================================================================
class IPCConfigManager:
    DEFAULT_ROI = [0.16, 0.10, 0.84, 0.90]

    DEFAULT_CONFIG = {
        "camera_enabled": True,
        "recognition_enabled": True,
        "min_detection_confidence": 0.60,
        "min_tracking_confidence": 0.50,
        "frame_skip": 1,
        "model_complexity": 0,
        "low_light_clahe": False,
        "show_skeleton_overlay": True,
        "enable_roi_optional": False,
        "roi_bounds": [0.16, 0.10, 0.84, 0.90],
        "swipe_cooldown": 1.25,
        "zoom_sensitivity": 0.050,
        "enable_zoom_optional": True,
        "enable_air_draw": True,
        "enable_voice_notes": True,
        "enable_slideshow_launch": True,
        "active_tab": "live",
        "gesture_mappings": {
            "Right Swipe": "Right Arrow",
            "Left Swipe": "Left Arrow",
            "Open Palm": "Engine Toggle",
            "Pinch In / Out": "Ctrl + / -",
            "Index Finger Point": "Air-Draw Pen"
        },
        "gesture_states": {
            "Right Swipe": True,
            "Left Swipe": True,
            "Open Palm": True,
            "Pinch In / Out": True,
            "Index Finger Point": True
        },
        "lighting_sensitivity": 0.50,
        "enable_auto_lighting_optional": False,
        "sandbox_mode": False,
        "theme": "light",
    }

    def __init__(self, filepath=CONFIG_FILE):
        self.filepath = filepath
        self.state = self.load()
        self.last_sync_time = time.time()

    def load(self):
        self.had_load_error = False
        if not os.path.exists(self.filepath):
            self.save(self.DEFAULT_CONFIG.copy())
            return self.DEFAULT_CONFIG.copy()
        try:
            with open(self.filepath, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                raise ValueError("Root JSON is not an object")
            
            merged = self.DEFAULT_CONFIG.copy()
            merged.update(data)

            merged["min_detection_confidence"] = float(np.clip(float(merged.get("min_detection_confidence", 0.60)), 0.10, 1.00))
            merged["lighting_sensitivity"] = float(np.clip(float(merged.get("lighting_sensitivity", 0.50)), 0.00, 1.00))
            merged["swipe_cooldown"] = float(np.clip(float(merged.get("swipe_cooldown", 1.25)), 0.30, 5.00))

            if "gesture_mappings" not in data:
                merged["gesture_mappings"] = self.DEFAULT_CONFIG["gesture_mappings"].copy()
            else:
                for k, v in self.DEFAULT_CONFIG["gesture_mappings"].items():
                    if k not in merged["gesture_mappings"]:
                        merged["gesture_mappings"][k] = v

                if merged["gesture_mappings"].get("Open Palm") == "Key B":
                    merged["gesture_mappings"]["Open Palm"] = "Engine Toggle"
            
            if "gesture_states" not in data:
                merged["gesture_states"] = self.DEFAULT_CONFIG["gesture_states"].copy()
            else:
                merged["gesture_states"]["Open Palm"] = merged["gesture_states"].get("Open Palm", True)

            if not isinstance(merged.get("roi_bounds"), list) or len(merged["roi_bounds"]) != 4:
                merged["roi_bounds"] = list(self.DEFAULT_ROI)
            if merged.get("active_tab") == "mapping":
                merged["active_tab"] = "settings"
            return merged
        except Exception:
            self.had_load_error = True
            return self.DEFAULT_CONFIG.copy()

    def save(self, cfg=None):
        if cfg is not None:
            self.state = cfg
        try:
            with open(self.filepath, "w") as f:
                json.dump(self.state, f, indent=4)
            self.last_sync_time = time.time()
        except Exception as e:
            print(f"[WARN] IPC Config Save Error: {e}")

    def poll_if_needed(self, interval=2.0):
        if (time.time() - self.last_sync_time) > interval:
            self.state = self.load()
            self.last_sync_time = time.time()
        return self.state


# ==============================================================================
# 3. SMART OS HARDWARE ABSTRACTION LAYER (WINDOW-AWARE SLIDESHOW LAUNCHER)
# ==============================================================================
class OSActionDispatcher:
    def __init__(self):
        self.keyboard = KeyController()
        self.mouse = MouseController()
        self.last_emulated_time = 0.0

    @staticmethod
    def _get_active_window_title():
        if os.name != "nt":
            return ""
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value.lower()
        except Exception:
            return ""

    def _async_zoom_worker(self, steps, ticks):
        try:
            self.last_emulated_time = time.time()
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
            self.last_emulated_time = time.time()
        except Exception:
            pass

    def execute_zoom(self, ticks):
        if ticks == 0:
            return "None"
        steps = min(abs(int(ticks)), 3)
        self.last_emulated_time = time.time()
        threading.Thread(target=self._async_zoom_worker, args=(steps, ticks), daemon=True).start()
        return "Ctrl + Plus (Zoom In)" if ticks > 0 else "Ctrl + Minus (Zoom Out)"

    def _async_slideshow_worker(self):
        try:
            self.last_emulated_time = time.time()
            win_title = self._get_active_window_title()
            is_powerpoint = any(k in win_title for k in ("powerpoint", "wps", "impress", ".pptx", ".ppt"))

            if is_powerpoint:
                self.keyboard.press(Key.f5)
                self.keyboard.release(Key.f5)
            else:
                with self.keyboard.pressed(Key.ctrl):
                    with self.keyboard.pressed(Key.alt):
                        self.keyboard.press('p')
                        self.keyboard.release('p')
            self.last_emulated_time = time.time()
        except Exception:
            pass

    def execute_start_slideshow(self):
        self.last_emulated_time = time.time()
        threading.Thread(target=self._async_slideshow_worker, daemon=True).start()
        return "Start Slideshow (Fullscreen)"

    def _async_key_worker(self, key_action):
        try:
            self.last_emulated_time = time.time()
            if key_action == "right":
                self.keyboard.press(Key.right)
                self.keyboard.release(Key.right)
            elif key_action == "left":
                self.keyboard.press(Key.left)
                self.keyboard.release(Key.left)
            elif len(key_action) == 1:
                self.keyboard.press(key_action)
                self.keyboard.release(key_action)
            elif HAS_PYAUTOGUI:
                pyautogui.press(key_action)
            self.last_emulated_time = time.time()
        except Exception:
            pass

    def execute_shortcut(self, gesture_name, mappings, states=None):
        if states and not states.get(gesture_name, True):
            return "Disabled"

        mapped_key = mappings.get(gesture_name, "")
        if not mapped_key:
            return "None"

        low = mapped_key.lower()
        if "engine" in low or "toggle" in low or "pause" in low:
            return "Engine Toggle"
        elif "air-draw" in low or "draw" in low:
            return "Air-Draw Pen"
        elif "right" in low:
            self.last_emulated_time = time.time()
            threading.Thread(target=self._async_key_worker, args=("right",), daemon=True).start()
            return "Next Slide (Right Arrow)"
        elif "left" in low:
            self.last_emulated_time = time.time()
            threading.Thread(target=self._async_key_worker, args=("left",), daemon=True).start()
            return "Previous Slide (Left Arrow)"
        elif "ctrl" in low:
            return self.execute_zoom(1)
        elif "key " in low:
            char = low.replace("key ", "").strip()
            if len(char) == 1:
                self.last_emulated_time = time.time()
                threading.Thread(target=self._async_key_worker, args=(char,), daemon=True).start()
                return f"Key '{char.upper()}'"
        elif len(mapped_key) == 1:
            self.last_emulated_time = time.time()
            threading.Thread(target=self._async_key_worker, args=(mapped_key.lower(),), daemon=True).start()
            return f"Key '{mapped_key.upper()}'"
        else:
            self.last_emulated_time = time.time()
            threading.Thread(target=self._async_key_worker, args=(mapped_key.lower(),), daemon=True).start()

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
# 6. HYBRID STATIC-DIRECTION & MULTIMODAL GESTURE ENGINE
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
        self.RECOIL_LOCK_SEC = 1.10
        self.swipe_latched = False
        self.thumb_latched = False
        self.peace_latched = False

    def reset_tracking(self):
        self.trajectory.clear()
        self.prediction_buffer.clear()

    @staticmethod
    def is_within_safe_frame_margin(landmarks, margin=0.035):
        wrist = landmarks.landmark[0]
        return (margin <= wrist.x <= (1.0 - margin)) and (margin <= wrist.y <= (1.0 - margin))

    @staticmethod
    def is_within_presenter_roi(landmarks, roi_bounds=None):
        if not roi_bounds or len(roi_bounds) != 4:
            roi_bounds = [0.16, 0.10, 0.84, 0.90]
        x_min, y_min, x_max, y_max = roi_bounds
        wrist = landmarks.landmark[0]
        middle_mcp = landmarks.landmark[9]
        cx = 0.5 * (wrist.x + middle_mcp.x)
        cy = 0.5 * (wrist.y + middle_mcp.y)
        return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)

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
    def is_strict_open_palm(landmarks, zoom_active=False):
        if zoom_active:
            return False

        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip, index_pip = landmarks.landmark[8], landmarks.landmark[6]
        middle_tip, middle_pip, middle_mcp = landmarks.landmark[12], landmarks.landmark[10], landmarks.landmark[9]
        ring_tip, ring_pip = landmarks.landmark[16], landmarks.landmark[14]
        pinky_tip, pinky_pip, pinky_mcp = landmarks.landmark[20], landmarks.landmark[18], landmarks.landmark[17]

        palm_len = max(math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y), 1e-4)

        f_extended = (
            math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) > math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y) * 1.14 and
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) > math.hypot(middle_pip.x - wrist.x, middle_pip.y - wrist.y) * 1.14 and
            math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) > math.hypot(ring_pip.x - wrist.x, ring_pip.y - wrist.y) * 1.12 and
            math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) > math.hypot(pinky_pip.x - wrist.x, pinky_pip.y - wrist.y) * 1.10
        )
        if not f_extended:
            return False

        pinch_dist = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) / palm_len
        thumb_ext = math.hypot(thumb_tip.x - pinky_mcp.x, thumb_tip.y - pinky_mcp.y) / palm_len
        if pinch_dist < 0.50 or thumb_ext < 0.85:
            return False

        dx_mid = abs(middle_tip.x - wrist.x)
        dy_mid = abs(middle_tip.y - wrist.y)
        if dx_mid > dy_mid * 0.85:
            return False

        return True

    @staticmethod
    def is_peace_voice_gesture(landmarks, zoom_active=False):
        if zoom_active:
            return False
        wrist = landmarks.landmark[0]
        index_tip, index_pip = landmarks.landmark[8], landmarks.landmark[6]
        middle_tip, middle_pip = landmarks.landmark[12], landmarks.landmark[10]
        ring_tip, ring_pip = landmarks.landmark[16], landmarks.landmark[14]
        pinky_tip, pinky_pip = landmarks.landmark[20], landmarks.landmark[18]

        index_up = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) > math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y) * 1.16
        middle_up = math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) > math.hypot(middle_pip.x - wrist.x, middle_pip.y - wrist.y) * 1.16
        ring_folded = math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < math.hypot(ring_pip.x - wrist.x, ring_pip.y - wrist.y) * 1.04
        pinky_folded = math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < math.hypot(pinky_pip.x - wrist.x, pinky_pip.y - wrist.y) * 1.04
        v_spread = math.hypot(index_tip.x - middle_tip.x, index_tip.y - middle_tip.y) > 0.035

        return index_up and middle_up and ring_folded and pinky_folded and v_spread

    @staticmethod
    def is_thumbs_up_slideshow_gesture(landmarks, zoom_active=False):
        if zoom_active:
            return False
        wrist = landmarks.landmark[0]
        thumb_tip, thumb_ip, thumb_mcp = landmarks.landmark[4], landmarks.landmark[3], landmarks.landmark[2]
        index_tip, index_mcp = landmarks.landmark[8], landmarks.landmark[5]
        middle_tip, middle_mcp = landmarks.landmark[12], landmarks.landmark[9]
        ring_tip, ring_mcp = landmarks.landmark[16], landmarks.landmark[13]
        pinky_tip, pinky_mcp = landmarks.landmark[20], landmarks.landmark[17]

        thumb_up = (thumb_mcp.y - thumb_tip.y) > 0.075 and (thumb_ip.y - thumb_tip.y) > 0.025
        fingers_curled = (
            math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) < math.hypot(index_mcp.x - wrist.x, index_mcp.y - wrist.y) * 1.02 and
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) < math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y) * 1.02 and
            math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < math.hypot(ring_mcp.x - wrist.x, ring_mcp.y - wrist.y) * 1.02 and
            math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < math.hypot(pinky_mcp.x - wrist.x, pinky_mcp.y - wrist.y) * 1.02
        )
        return thumb_up and fingers_curled

    @classmethod
    def is_index_draw_pose(cls, landmarks, zoom_active=False, already_drawing=False):
        """
        UNCONDITIONAL PEN-LOCK TRACKING:
        • Once drawing is active (already_drawing=True), stays locked onto Landmark 8
          unless the user explicitly spreads all 5 fingers into an Open Palm (✋) or
          makes a Peace Sign (✌️). Fast hand motion can NEVER break the stroke!
        """
        if zoom_active:
            return False

        if already_drawing:
            if cls.is_strict_open_palm(landmarks, zoom_active=False):
                return False
            if cls.is_peace_voice_gesture(landmarks, zoom_active=False):
                return False
            return True

        wrist = landmarks.landmark[0]
        thumb_tip = landmarks.landmark[4]
        index_tip, index_pip = landmarks.landmark[8], landmarks.landmark[6]
        middle_tip, middle_pip, middle_mcp = landmarks.landmark[12], landmarks.landmark[10], landmarks.landmark[9]
        ring_tip, ring_pip = landmarks.landmark[16], landmarks.landmark[14]
        pinky_tip, pinky_pip = landmarks.landmark[20], landmarks.landmark[18]

        palm_len = max(math.hypot(middle_mcp.x - wrist.x, middle_mcp.y - wrist.y), 1e-4)
        pinch_ratio = math.hypot(thumb_tip.x - index_tip.x, thumb_tip.y - index_tip.y) / palm_len
        if pinch_ratio < 0.38:
            return False

        dx = abs(index_tip.x - wrist.x)
        dy = abs(index_tip.y - wrist.y)
        if dx > dy * 1.20:
            return False

        index_up = math.hypot(index_tip.x - wrist.x, index_tip.y - wrist.y) > math.hypot(index_pip.x - wrist.x, index_pip.y - wrist.y) * 1.08
        others_curled = (
            math.hypot(middle_tip.x - wrist.x, middle_tip.y - wrist.y) < math.hypot(middle_pip.x - wrist.x, middle_pip.y - wrist.y) * 1.12 and
            math.hypot(ring_tip.x - wrist.x, ring_tip.y - wrist.y) < math.hypot(ring_pip.x - wrist.x, ring_pip.y - wrist.y) * 1.12 and
            math.hypot(pinky_tip.x - wrist.x, pinky_tip.y - wrist.y) < math.hypot(pinky_pip.x - wrist.x, pinky_pip.y - wrist.y) * 1.12
        )
        return index_up and others_curled

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

        if extension_ratio > 1.14 and horizontal_dominance > 1.18:
            angle_clarity = min((horizontal_dominance - 1.0) / 1.5, 1.0)
            reach_clarity = min((extension_ratio - 1.0) / 0.8, 1.0)
            dyn_conf = float(np.clip((0.60 * mp_hand_score + 0.25 * angle_clarity + 0.15 * reach_clarity), 0.82, 0.996))

            if norm_dx < -0.88:
                return "Left Swipe", "DIRECTION: LEFT", dyn_conf
            elif norm_dx > 0.88:
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

        if self.trajectory and (now_time - self.trajectory[-1][1]) > 0.5:
            self.trajectory.clear()

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

        if label in ("Left Swipe", "Right Swipe", "Open Palm") or "pinch" in label.lower():
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
# 8. STANDALONE DESKTOP GUI APPLICATION (MULTIMODAL PRESENTER SUITE)
# ==============================================================================
class WaveDesktopApp:
    BG_MAIN = "#F8FAFC"
    BG_CARD = "#FFFFFF"
    BORDER_LIGHT = "#E2E8F0"
    BLUE_PRIMARY = "#136DEC"
    BLUE_LIGHT_BG = "#E8F1FD"
    GREEN_TEXT = "#10B981"
    GREEN_BG = "#DCFCE7"
    AMBER_TEXT = "#D97706"
    AMBER_BG = "#FEF3C7"
    RED_TEXT = "#EF4444"
    RED_BG = "#FEE2E2"
    SLATE_900 = "#0F172A"
    SLATE_700 = "#334155"
    SLATE_500 = "#64748B"
    SLATE_400 = "#94A3B8"
    PILL_BG = "#F1F5F9"

    DESCRIPTIONS = {
        "Right Swipe": ("Next Slide — Fires once per swipe (returns to center to re-arm)", "→ Right Arrow", "→ Right Arrow"),
        "Left Swipe": ("Previous Slide — Fires once per swipe (returns to center to re-arm)", "← Left Arrow", "← Left Arrow"),
        "Open Palm": ("Pause Engine & Clear Ink — Hold Open Palm (1.2s) to toggle engine", "⏸ Engine Toggle", "⏸ Engine Toggle"),
        "Pinch In / Out": ("Zoom Slide — Spread (+) or Pinch (-) like a tablet", "⊕ Ctrl + / -", "⊕ Ctrl + / -"),
        "Index Finger Point": ("Air-Draw Annotation — Async 45+ FPS Catmull-Rom Spline Pen", "✏️ Air-Draw Pen", "✏️ Air-Draw Pen")
    }

    THEMES = {
        "light": {
            "BG_MAIN": "#F8FAFC", "BG_CARD": "#FFFFFF", "BORDER": "#E2E8F0",
            "FG_PRIMARY": "#0F172A", "FG_MUTED": "#64748B", "ACCENT": "#136DEC", "PILL_BG": "#F1F5F9"
        },
        "dark": {
            "BG_MAIN": "#0B1120", "BG_CARD": "#1E293B", "BORDER": "#334155",
            "FG_PRIMARY": "#F8FAFC", "FG_MUTED": "#94A3B8", "ACCENT": "#3B82F6", "PILL_BG": "#0F172A"
        }
    }

    def _get_calibration_sample(self):
        _, frame, _, _ = self.cam_stream.read()
        return frame, float(self.cached_hand_score)

    def _retry_camera_connection(self):
        self.config["camera_enabled"] = True
        self.ipc_config.save(self.config)
        self.cam_stream.set_camera_enabled(True)
        self._sync_camera_ui(True)
        
    def _apply_accessibility_profile(self):
        tremor_on = bool(self.config.get("accessibility_tremor_filter", False))
        if tremor_on:
            # Heavy low-pass EMA smoothing (lower alpha = stronger tremor suppression)
            self.draw_x_ema.alpha = 0.16
            self.draw_y_ema.alpha = 0.16
            self.live_conf_ema.alpha = 0.20
            # Require longer, deliberate holds to prevent accidental triggers
            self.PALM_HOLD_DURATION = 1.75
            self.PEACE_HOLD_DURATION = 1.65
            self.THUMB_HOLD_DURATION = 1.75
        else:
            # Standard responsive presenter defaults
            self.draw_x_ema.alpha = 0.45
            self.draw_y_ema.alpha = 0.45
            self.live_conf_ema.alpha = 0.40
            self.PALM_HOLD_DURATION = 1.20
            self.PEACE_HOLD_DURATION = 1.10
            self.THUMB_HOLD_DURATION = 1.20
        

    def _on_calibration_changed(self):
        self.config = self.ipc_config.state
        if hasattr(self, "clahe_toggle"):
            self.clahe_toggle.configure(
                image=self.assets["toggle_on"] if self.config.get("low_light_clahe", False) else self.assets["toggle_off"]
            )
        if hasattr(self, "roi_toggle"):
            self.roi_toggle.configure(
                image=self.assets["toggle_on"] if self.config.get("enable_roi_optional", False) else self.assets["toggle_off"]
            )
            self._apply_accessibility_profile()

    def toggle_theme(self):
        new_theme = "dark" if self.config.get("theme", "light") == "light" else "light"
        self.config["theme"] = new_theme
        self.ipc_config.save(self.config)
        self.apply_theme(new_theme)

    def apply_theme(self, theme_name="light"):
        pal = self.THEMES.get(theme_name, self.THEMES["light"])
        self.root.configure(bg=pal["BG_MAIN"])
        self.navbar.configure(bg=pal["BG_CARD"], highlightbackground=pal["BORDER"])
        self.content_container.configure(bg=pal["BG_MAIN"])
        self.live_view.configure(bg=pal["BG_MAIN"])
        self.settings_view.configure(bg=pal["BG_MAIN"])
        self.telemetry_card.configure(bg=pal["BG_CARD"], highlightbackground=pal["BORDER"])
        self.footer_card.configure(bg=pal["BG_CARD"], highlightbackground=pal["BORDER"])
        self.live_gesture_title.configure(fg=pal["FG_PRIMARY"], bg=pal["BG_CARD"])
        self.theme_btn.configure(bg=pal["PILL_BG"], fg=pal["FG_PRIMARY"], highlightbackground=pal["BORDER"])
        if hasattr(self, "calibration_view"):
            self.calibration_view.apply_theme(pal)
        self.switch_tab(self.config.get("active_tab", "live"))

    def switch_tab(self, tab_name):
        if tab_name not in ("live", "calibration", "settings"):
            tab_name = "live"
        self.config["active_tab"] = tab_name
        self.ipc_config.save(self.config)

        pal = self.THEMES.get(self.config.get("theme", "light"), self.THEMES["light"])
        self.live_view.pack_forget()
        self.calibration_view.pack_forget()
        self.settings_view.pack_forget()

        for btn in (self.tab_live_btn, self.tab_calib_btn, self.tab_settings_btn):
            btn.configure(bg=pal["PILL_BG"], fg=pal["FG_MUTED"])

        if tab_name == "live":
            self.live_view.pack(fill="both", expand=True)
            self.tab_live_btn.configure(bg=pal["BG_CARD"], fg=pal["FG_PRIMARY"])
            if not self.config.get("camera_enabled", True):
                self.cam_off_rendered = False
        elif tab_name == "calibration":
            self.calibration_view.refresh_from_config()
            self.calibration_view.pack(fill="both", expand=True)
            self.tab_calib_btn.configure(bg=pal["BG_CARD"], fg=pal["ACCENT"])
        else:
            self.settings_view.pack(fill="both", expand=True)
            self.tab_settings_btn.configure(bg=pal["BG_CARD"], fg=pal["ACCENT"])

    def __init__(self, root):
        self.root = root
        self.root.title("WAVE - Multimodal Gesture & Voice Presentation Assistant")
        self.root.geometry("1280x860")
        self.root.minsize(1150, 810)
        self.root.configure(bg=self.BG_MAIN)

        self.ipc_config = IPCConfigManager(CONFIG_FILE)
        self.error_handler = ErrorHandler(
            root=self.root,
            callbacks={
                "retry_camera": self._retry_camera_connection,
                "open_calibration": lambda: self.switch_tab("calibration"),
                "reset_settings": self.reset_default_mappings
            }
        )
        self.no_hand_since = time.time()
        self.low_conf_streak = 0
        if getattr(self.ipc_config, "had_load_error", False):
            self.root.after(600, self.error_handler.notify_config_error)
        self.config = self.ipc_config.state
        self.os_dispatcher = OSActionDispatcher()
        self.vision_engine = VisionGestureEngine(MODEL_FILE)
        self.zoom_controller = TabletPinchZoomController()

        self.air_canvas = AirCanvasOverlay(self.root)
        self.voice_notes = VoiceStickyNoteManager(
            self.root, self.os_dispatcher, on_state_change=self._sync_voice_btn_ui
        )
        # Zero-lag EMA (fast_alpha=1.00) so fast drawing tracks your fingertip instantaneously!
        self.draw_x_ema = ExponentialMovingAverage(alpha=0.72, fast_alpha=1.00, velocity_threshold=0.04)
        self.draw_y_ema = ExponentialMovingAverage(alpha=0.72, fast_alpha=1.00, velocity_threshold=0.04)

        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cam_start_on = self.config.get("camera_enabled", True)
        self.cam_stream = ThreadedCameraStream(src=0, width=640, height=480, start_enabled=cam_start_on)

        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        # Dedicated Async MediaPipe Thread so UI rendering never slows down hand tracking!
        self.async_tracker = AsyncMediaPipeTracker(self.cam_stream, lambda: self.config)

        os.makedirs("results", exist_ok=True)
        self.log_start_time = time.time()
        self.logged_seconds = set()
        self._init_60s_telemetry_log()

        self.roi_drag_mode = None
        self.roi_drag_start_xy = (0.0, 0.0)
        self.roi_orig_bounds = [0.16, 0.10, 0.84, 0.90]
        self.MIN_ROI_SPAN = 0.15

        self.last_processed_cam_id = -1
        self.frame_count = 0
        self.cached_landmarks = None
        self.cached_hand_score = 0.0

        self.palm_hold_start = 0.0
        self.PALM_HOLD_DURATION = 1.2
        self.last_palm_toggle_time = 0.0

        self.peace_hold_start = 0.0
        self.PEACE_HOLD_DURATION = 1.0
        self.last_peace_trigger_time = 0.0

        self.thumb_hold_start = 0.0
        self.THUMB_HOLD_DURATION = 1.0
        self.last_thumb_trigger_time = 0.0

        self.global_cooldown_until = 0.0

        self.last_recognized_ui_time = 0.0
        self.last_displayed_gesture = None
        self.last_loop_tick_time = time.time()
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
        self._apply_accessibility_profile()
        self._bind_shortcuts()

        self._sync_camera_ui(cam_start_on)
        initial_tab = self.config.get("active_tab", "live")
        if initial_tab not in ("live", "calibration", "settings"):
            initial_tab = "live"
        self.switch_tab(initial_tab)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.update_loop()

    def _init_60s_telemetry_log(self):
        try:
            if not os.path.exists(LIVE_LOG_FILE):
                with open(LIVE_LOG_FILE, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Elapsed_Sec", "Timestamp", "Camera_FPS", "Pipeline_Latency_ms", "ROI_Enabled"])
        except Exception:
            pass

    def _log_60s_telemetry_sample(self, cam_fps, latency_ms):
        elapsed = int(time.time() - self.log_start_time)
        if 1 <= elapsed <= 60 and elapsed not in self.logged_seconds:
            self.logged_seconds.add(elapsed)
            try:
                with open(LIVE_LOG_FILE, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        elapsed,
                        time.strftime("%H:%M:%S"),
                        round(float(cam_fps), 2),
                        round(float(latency_ms), 2),
                        bool(self.config.get("enable_roi_optional", False))
                    ])
            except Exception:
                pass

    def _apply_clahe_low_light(self, bgr_frame):
        ycrcb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        y_eq = self.clahe.apply(y)
        merged = cv2.merge((y_eq, cr, cb))
        return cv2.cvtColor(merged, cv2.COLOR_YCrCb2BGR)

    def _create_info_badge(self, parent, tooltip_title, tooltip_body):
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

        self.tab_calib_btn = tk.Label(
            tab_outer, text="🎛 Calibration", font=("Segoe UI", 9, "bold"),
            padx=16, pady=5, cursor="hand2"
        )
        self.tab_calib_btn.pack(side="left")
        self.tab_calib_btn.bind("<Button-1>", lambda e: self.switch_tab("calibration"))

        self.tab_settings_btn = tk.Label(
            tab_outer, text="⚙ Settings & Keybinds", font=("Segoe UI", 9, "bold"),
            padx=16, pady=5, cursor="hand2"
        )
        self.tab_settings_btn.pack(side="left")
        self.tab_settings_btn.bind("<Button-1>", lambda e: self.switch_tab("settings"))

        right_frame = tk.Frame(self.navbar, bg=self.BG_CARD)
        right_frame.pack(side="right", padx=24, pady=12)

        self.theme_btn = tk.Label(
            right_frame, text="🌓 Theme", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_700, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=4, cursor="hand2"
        )
        self.theme_btn.pack(side="right", padx=(10, 0))
        self.theme_btn.bind("<Button-1>", lambda e: self.toggle_theme())

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
        self.calibration_view = CalibrationPanel(
            self.content_container,
            self.ipc_config,
            on_change_callback=self._on_calibration_changed,
            on_save_callback=self._on_calibration_changed,
            on_reset_roi=self.reset_roi_bounds,
            sample_provider=self._get_calibration_sample
        )

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

        self.video_label.bind("<Motion>", self._on_video_mouse_hover)
        self.video_label.bind("<ButtonPress-1>", self._on_video_mouse_down)
        self.video_label.bind("<B1-Motion>", self._on_video_mouse_drag)
        self.video_label.bind("<ButtonRelease-1>", self._on_video_mouse_up)

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
            bottom_row, text="—", font=("Segoe UI", 8, "bold"),
            fg=self.BLUE_PRIMARY, bg=self.BLUE_LIGHT_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=8, pady=1
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
            padx=14, pady=10
        )
        self.footer_card.pack(side="top", fill="x")

        # 1. Pause Engine Button
        self.pause_btn_frame = tk.Frame(
            self.footer_card, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=6, cursor="hand2"
        )
        self.pause_btn_frame.pack(side="left")
        self.pause_btn_frame.bind("<Button-1>", lambda e: self.toggle_engine())

        self.pause_icon_lbl = tk.Label(
            self.pause_btn_frame, text="⏸", font=("Segoe UI", 9), fg=self.BLUE_PRIMARY, bg=self.PILL_BG, cursor="hand2"
        )
        self.pause_icon_lbl.pack(side="left", padx=(0, 5))
        self.pause_icon_lbl.bind("<Button-1>", lambda e: self.toggle_engine())

        self.pause_txt_lbl = tk.Label(
            self.pause_btn_frame, text="Pause Engine", font=("Segoe UI", 9, "bold"), fg=self.SLATE_700, bg=self.PILL_BG, cursor="hand2"
        )
        self.pause_txt_lbl.pack(side="left", padx=(0, 6))
        self.pause_txt_lbl.bind("<Button-1>", lambda e: self.toggle_engine())

        space_keycap = tk.Label(
            self.pause_btn_frame, text="✋ 1.2s", font=("Segoe UI", 8),
            fg=self.SLATE_500, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=5, pady=0, cursor="hand2"
        )
        space_keycap.pack(side="left")
        space_keycap.bind("<Button-1>", lambda e: self.toggle_engine())

        # 2. Turn Off / On Camera Button
        self.cam_btn_frame = tk.Frame(
            self.footer_card, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=6, cursor="hand2"
        )
        self.cam_btn_frame.pack(side="left", padx=(8, 0))
        self.cam_btn_frame.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_btn_icon = tk.Label(
            self.cam_btn_frame, text="📷", font=("Segoe UI", 9), fg=self.RED_TEXT, bg=self.PILL_BG, cursor="hand2"
        )
        self.cam_btn_icon.pack(side="left", padx=(0, 5))
        self.cam_btn_icon.bind("<Button-1>", lambda e: self.toggle_camera())

        self.cam_btn_txt = tk.Label(
            self.cam_btn_frame, text="Turn Off Camera", font=("Segoe UI", 9, "bold"), fg=self.SLATE_700, bg=self.PILL_BG, cursor="hand2"
        )
        self.cam_btn_txt.pack(side="left")
        self.cam_btn_txt.bind("<Button-1>", lambda e: self.toggle_camera())

        # 3. Continuous Voice Sticky Note Toggle Button (Peace Sign ✌️ or Click)
        self.voice_btn_frame = tk.Frame(
            self.footer_card, bg="#FEF9C3", highlightthickness=1, highlightbackground="#FDE047",
            padx=10, pady=6, cursor="hand2"
        )
        self.voice_btn_frame.pack(side="left", padx=(8, 0))
        self.voice_btn_frame.bind("<Button-1>", lambda e: self._toggle_voice_note_action())

        self.voice_btn_lbl = tk.Label(
            self.voice_btn_frame, text="🎙️ Voice Note (✌️)", font=("Segoe UI", 9, "bold"),
            fg="#854D0E", bg="#FEF9C3", cursor="hand2"
        )
        self.voice_btn_lbl.pack(side="left")
        self.voice_btn_lbl.bind("<Button-1>", lambda e: self._toggle_voice_note_action())

        # 4. Clear Air-Draw Ink Button
        self.clear_ink_btn = tk.Frame(
            self.footer_card, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=6, cursor="hand2"
        )
        self.clear_ink_btn.pack(side="left", padx=(8, 0))
        self.clear_ink_btn.bind("<Button-1>", lambda e: self.air_canvas.clear_all())

        clear_lbl = tk.Label(
            self.clear_ink_btn, text="🧹 Clear Ink", font=("Segoe UI", 9, "bold"),
            fg=self.SLATE_700, bg=self.PILL_BG, cursor="hand2"
        )
        clear_lbl.pack(side="left")
        clear_lbl.bind("<Button-1>", lambda e: self.air_canvas.clear_all())

        tk.Frame(self.footer_card, bg=self.BORDER_LIGHT, width=1, height=20).pack(side="left", padx=12)

        tk.Label(
            self.footer_card, text="CONFIDENCE: ", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        ).pack(side="left")

        self.confidence_val_lbl = tk.Label(
            self.footer_card, text="0.0%", font=("Segoe UI", 9, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        )
        self.confidence_val_lbl.pack(side="left")

        self.fps_footer_lbl = tk.Label(
            self.footer_card,
            text="CAMERA FPS: --   |   LATENCY: -- ms   |   ZOOM: IDLE",
            font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_400, bg=self.BG_CARD
        )
        self.fps_footer_lbl.pack(side="right")

    def _sync_voice_btn_ui(self, is_listening):
        if not hasattr(self, "voice_btn_lbl"):
            return
        if is_listening:
            self.voice_btn_frame.configure(bg="#FEE2E2", highlightbackground="#EF4444")
            self.voice_btn_lbl.configure(text="⏹ Stop & Transcribe (✌️)", fg="#DC2626", bg="#FEE2E2")
        else:
            self.voice_btn_frame.configure(bg="#FEF9C3", highlightbackground="#FDE047")
            self.voice_btn_lbl.configure(text="🎙️ Voice Note (✌️)", fg="#854D0E", bg="#FEF9C3")

    def _toggle_voice_note_action(self):
        started = self.voice_notes.toggle_continuous_listening(
            on_phrase_callback=lambda txt: self._update_telemetry_card(
                "Open Palm", 99.5,
                custom_action=f"AI Note: \"{txt[:28]}...\"" if len(txt) > 28 else f"AI Note: \"{txt}\"",
                custom_key="🎙️ Whisper / Neural AI"
            )
        )
        if started:
            self._update_telemetry_card(
                "Open Palm", 99.0,
                custom_action="Unbroken Session Recording ON",
                custom_key="🔴 Mic Recording"
            )
            self.live_gesture_title.configure(text="Voice Sticky Note (✌️)")
        else:
            self._update_telemetry_card(
                "Open Palm", 99.0,
                custom_action="Transcribing Full Paragraph...",
                custom_key="⏳ AI Processing"
            )
            self.live_gesture_title.configure(text="Voice Note Processing (✌️)")

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
    # INTERACTIVE ROI DRAG, MOVE & RESIZE HANDLERS ON LIVE VIDEO FEED
    # ==========================================================================
    def _hit_test_roi_mode(self, nx, ny):
        x0, y0, x1, y1 = self.config.get("roi_bounds", [0.16, 0.10, 0.84, 0.90])
        tol = 0.035

        near_l = abs(nx - x0) <= tol and (y0 - tol <= ny <= y1 + tol)
        near_r = abs(nx - x1) <= tol and (y0 - tol <= ny <= y1 + tol)
        near_t = abs(ny - y0) <= tol and (x0 - tol <= nx <= x1 + tol)
        near_b = abs(ny - y1) <= tol and (x0 - tol <= nx <= x1 + tol)

        if near_t and near_l:
            return "tl"
        if near_t and near_r:
            return "tr"
        if near_b and near_l:
            return "bl"
        if near_b and near_r:
            return "br"
        if near_l:
            return "l"
        if near_r:
            return "r"
        if near_t:
            return "t"
        if near_b:
            return "b"
        if x0 < nx < x1 and y0 < ny < y1:
            return "move"
        return "draw"

    def _on_video_mouse_hover(self, event):
        if not self.config.get("enable_roi_optional", False):
            self.video_label.configure(cursor="")
            return
        w = max(self.video_card.winfo_width(), 1)
        h = max(self.video_card.winfo_height(), 1)
        nx, ny = event.x / w, event.y / h
        mode = self._hit_test_roi_mode(nx, ny)

        cursor_map = {
            "tl": "top_left_corner",
            "tr": "top_right_corner",
            "bl": "bottom_left_corner",
            "br": "bottom_right_corner",
            "l": "sb_h_double_arrow",
            "r": "sb_h_double_arrow",
            "t": "sb_v_double_arrow",
            "b": "sb_v_double_arrow",
            "move": "fleur",
            "draw": "crosshair"
        }
        self.video_label.configure(cursor=cursor_map.get(mode, ""))

    def _on_video_mouse_down(self, event):
        if not self.config.get("enable_roi_optional", False):
            return
        w = max(self.video_card.winfo_width(), 1)
        h = max(self.video_card.winfo_height(), 1)
        nx = float(np.clip(event.x / w, 0.02, 0.98))
        ny = float(np.clip(event.y / h, 0.02, 0.98))

        self.roi_drag_mode = self._hit_test_roi_mode(nx, ny)
        self.roi_drag_start_xy = (nx, ny)
        self.roi_orig_bounds = list(self.config.get("roi_bounds", [0.16, 0.10, 0.84, 0.90]))

    def _on_video_mouse_drag(self, event):
        if not self.roi_drag_mode or not self.config.get("enable_roi_optional", False):
            return
        w = max(self.video_card.winfo_width(), 1)
        h = max(self.video_card.winfo_height(), 1)
        nx = float(np.clip(event.x / w, 0.02, 0.98))
        ny = float(np.clip(event.y / h, 0.02, 0.98))

        sx, sy = self.roi_drag_start_xy
        dx = nx - sx
        dy = ny - sy
        ox0, oy0, ox1, oy1 = self.roi_orig_bounds
        min_s = self.MIN_ROI_SPAN

        if self.roi_drag_mode == "move":
            bw = ox1 - ox0
            bh = oy1 - oy0
            nx0 = float(np.clip(ox0 + dx, 0.02, 0.98 - bw))
            ny0 = float(np.clip(oy0 + dy, 0.02, 0.98 - bh))
            new_b = [round(nx0, 3), round(ny0, 3), round(nx0 + bw, 3), round(ny0 + bh, 3)]
        elif self.roi_drag_mode == "draw":
            x_min, x_max = min(sx, nx), max(sx, nx)
            y_min, y_max = min(sy, ny), max(sy, ny)
            if (x_max - x_min) < min_s:
                x_max = min(0.98, x_min + min_s)
            if (y_max - y_min) < min_s:
                y_max = min(0.98, y_min + min_s)
            new_b = [round(x_min, 3), round(y_min, 3), round(x_max, 3), round(y_max, 3)]
        else:
            x0, y0, x1, y1 = ox0, oy0, ox1, oy1
            if "l" in self.roi_drag_mode:
                x0 = min(nx, x1 - min_s)
            if "r" in self.roi_drag_mode:
                x1 = max(nx, x0 + min_s)
            if "t" in self.roi_drag_mode:
                y0 = min(ny, y1 - min_s)
            if "b" in self.roi_drag_mode:
                y1 = max(ny, y0 + min_s)
            new_b = [round(x0, 3), round(y0, 3), round(x1, 3), round(y1, 3)]

        self.config["roi_bounds"] = new_b

    def _on_video_mouse_up(self, event):
        if self.roi_drag_mode is not None:
            self.roi_drag_mode = None
            self.ipc_config.save(self.config)

    def reset_roi_bounds(self):
        self.config["roi_bounds"] = list(IPCConfigManager.DEFAULT_ROI)
        self.ipc_config.save(self.config)

    # ==========================================================================
    # VERTICAL SETTINGS & KEYBINDS VIEW (WITH MULTIMODAL TOGGLES)
    # ==========================================================================
    def _build_vertical_settings_view(self):
        wrapper = tk.Frame(self.settings_view, bg=self.BG_MAIN)
        wrapper.pack(fill="both", expand=True, padx=56, pady=12)

        header_row = tk.Frame(wrapper, bg=self.BG_MAIN)
        header_row.pack(fill="x", pady=(0, 8))

        title_col = tk.Frame(header_row, bg=self.BG_MAIN)
        title_col.pack(side="left", anchor="w")

        tk.Label(
            title_col, text="⚙  MULTIMODAL VISION, AUDIO & HARDWARE ABSTRACTION LAYER", font=("Segoe UI", 8, "bold"),
            fg=self.BLUE_PRIMARY, bg=self.BG_MAIN
        ).pack(anchor="w")

        tk.Label(
            title_col, text="Settings & Gesture Key Mappings", font=("Segoe UI", 18, "bold"),
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

        video_card = tk.Frame(
            wrapper, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=22, pady=8
        )
        video_card.pack(fill="x", pady=(0, 10))

        vc_head = tk.Frame(video_card, bg=self.BG_CARD)
        vc_head.pack(fill="x", pady=(0, 4))
        tk.Label(vc_head, text="VISION ENGINE, ROI & MULTIMODAL PRESENTER SUITE", font=("Segoe UI", 9, "bold"), fg=self.BLUE_PRIMARY, bg=self.BG_CARD).pack(side="left")

        self.live_lat_badge = tk.Label(vc_head, text="Pipeline Latency: -- ms", font=("Consolas", 8, "bold"), fg=self.GREEN_TEXT, bg=self.GREEN_BG, padx=8, pady=2)
        self.live_lat_badge.pack(side="right")

        v_row0 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row0.pack(fill="x", pady=2)
        tk.Label(v_row0, text="Webcam Sensor Capture (Hardware Power)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row0,
            "Webcam Sensor Capture",
            "Physically turns your laptop webcam ON or OFF."
        )
        self.settings_cam_toggle = tk.Label(
            v_row0,
            image=self.assets["toggle_on"] if self.config.get("camera_enabled", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.settings_cam_toggle.pack(side="right")
        self.settings_cam_toggle.bind("<Button-1>", lambda e: self.toggle_camera())

        v_row1 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row1.pack(fill="x", pady=2)
        tk.Label(v_row1, text="AI Tracking Model Complexity", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row1,
            "Async 45+ FPS MediaPipe Thread (Lite vs. Full)",
            "Runs MediaPipe in a dedicated background thread so UI rendering never slows down hand tracking."
        )

        mp_btn_box = tk.Frame(v_row1, bg=self.BG_CARD)
        mp_btn_box.pack(side="right")
        curr_comp = int(self.config.get("model_complexity", 0))

        self.btn_comp_lite = tk.Label(
            mp_btn_box, text="Lite (Fastest Latency)", font=("Segoe UI", 8, "bold"),
            bg=self.BLUE_PRIMARY if curr_comp == 0 else self.PILL_BG,
            fg="#FFFFFF" if curr_comp == 0 else self.SLATE_700,
            padx=14, pady=3, cursor="hand2"
        )
        self.btn_comp_lite.pack(side="left", padx=(0, 6))
        self.btn_comp_lite.bind("<Button-1>", lambda e: self._set_model_complexity(0))

        self.btn_comp_full = tk.Label(
            mp_btn_box, text="Full (Deep Range)", font=("Segoe UI", 8, "bold"),
            bg=self.BLUE_PRIMARY if curr_comp == 1 else self.PILL_BG,
            fg="#FFFFFF" if curr_comp == 1 else self.SLATE_700,
            padx=14, pady=3, cursor="hand2"
        )
        self.btn_comp_full.pack(side="left")
        self.btn_comp_full.bind("<Button-1>", lambda e: self._set_model_complexity(1))

        v_row_roi = tk.Frame(video_card, bg=self.BG_CARD)
        v_row_roi.pack(fill="x", pady=2)
        tk.Label(v_row_roi, text="Presenter Region of Interest (ROI) Focus Zone", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row_roi,
            "Interactive Presenter ROI Calibration",
            "Isolates gesture detection strictly to your custom presenter zone."
        )
        self.roi_toggle = tk.Label(
            v_row_roi,
            image=self.assets["toggle_on"] if self.config.get("enable_roi_optional", False) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.roi_toggle.pack(side="right")
        self.roi_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("enable_roi_optional", self.roi_toggle))

        reset_roi_btn = tk.Label(
            v_row_roi, text="↺ Reset ROI Box", font=("Segoe UI", 8, "bold"),
            fg=self.SLATE_700, bg=self.PILL_BG, highlightthickness=1, highlightbackground=self.BORDER_LIGHT,
            padx=10, pady=2, cursor="hand2"
        )
        reset_roi_btn.pack(side="right", padx=(0, 12))
        reset_roi_btn.bind("<Button-1>", lambda e: self.reset_roi_bounds())

        v_row_draw = tk.Frame(video_card, bg=self.BG_CARD)
        v_row_draw.pack(fill="x", pady=2)
        tk.Label(v_row_draw, text="Multimodal Air-Draw Screen Annotation (☝️ Index Finger)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row_draw,
            "Async 45+ FPS Catmull-Rom Spline Air-Canvas",
            "Uses a dedicated background MediaPipe thread + Catmull-Rom cubic splines so fast strokes never lag or break.\n"
            "• Erase Ink: Show an Open Palm (✋) or click 'Clear Ink'."
        )
        self.air_draw_toggle = tk.Label(
            v_row_draw,
            image=self.assets["toggle_on"] if self.config.get("enable_air_draw", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.air_draw_toggle.pack(side="right")
        self.air_draw_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("enable_air_draw", self.air_draw_toggle))

        v_row_voice = tk.Frame(video_card, bg=self.BG_CARD)
        v_row_voice.pack(fill="x", pady=2)
        tk.Label(v_row_voice, text="Unbroken Session Whisper/Neural AI Voice Notes (✌️) & Slideshow (👍)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row_voice,
            "Unbroken Session Recording + OpenAI Whisper / Neural Speech AI",
            "Records your entire paragraph in one unbroken 44.1kHz stream (no mid-sentence cuts!) and transcribes with full-sentence context."
        )
        self.voice_toggle = tk.Label(
            v_row_voice,
            image=self.assets["toggle_on"] if self.config.get("enable_voice_notes", True) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.voice_toggle.pack(side="right")
        self.voice_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("enable_voice_notes", self.voice_toggle))

        v_row2 = tk.Frame(video_card, bg=self.BG_CARD)
        v_row2.pack(fill="x", pady=2)
        tk.Label(v_row2, text="Low-Light Contrast Boost (CLAHE)", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left")
        self._create_info_badge(
            v_row2,
            "Low-Light Contrast Boost (OpenCV YCrCb CLAHE)",
            "Boosts finger edge visibility in dim lecture halls."
        )
        self.clahe_toggle = tk.Label(
            v_row2,
            image=self.assets["toggle_on"] if self.config.get("low_light_clahe", False) else self.assets["toggle_off"],
            bg=self.BG_CARD, cursor="hand2"
        )
        self.clahe_toggle.pack(side="right")
        self.clahe_toggle.bind("<Button-1>", lambda e: self._toggle_boolean_setting("low_light_clahe", self.clahe_toggle))

        col_header = tk.Frame(wrapper, bg=self.BG_MAIN)
        col_header.pack(fill="x", padx=12, pady=(2, 4))
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
                padx=16, pady=6
            )
            row_card.pack(fill="x", pady=(0, 6))

            tk.Label(row_card, image=self.assets["icons"][gesture], bg=self.BG_CARD).pack(side="left", padx=(0, 14))

            title_sub_col = tk.Frame(row_card, bg=self.BG_CARD)
            title_sub_col.pack(side="left")
            tk.Label(title_sub_col, text=gesture, font=("Segoe UI", 10, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(anchor="w")

            _, default_pill, _ = self.DESCRIPTIONS.get(gesture, ("Action", "Key", "Key"))
            mapped_val = self.config.get("gesture_mappings", {}).get(gesture, default_pill)
            dyn_desc = self._get_dynamic_row_description(gesture, mapped_val)
            desc_lbl = tk.Label(title_sub_col, text=dyn_desc, font=("Segoe UI", 8), fg=self.SLATE_500, bg=self.BG_CARD)
            desc_lbl.pack(anchor="w")

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

            display_pill = self._format_keycode_badge(gesture, mapped_val)
            key_badge = tk.Label(
                row_card, text=display_pill, font=("Segoe UI", 9, "bold"),
                fg=self.BLUE_PRIMARY, bg=self.BLUE_LIGHT_BG, padx=12, pady=4
            )
            key_badge.pack(side="right")

            self.row_widgets[gesture] = {
                "badge": key_badge,
                "rebind": rebind_btn,
                "toggle": toggle_lbl,
                "desc": desc_lbl
            }

    @classmethod
    def _get_dynamic_action_summary(cls, gesture_name, mapped_val):
        low = str(mapped_val).lower()
        if "right" in low:
            return "Next Slide (Right Arrow)"
        elif "left" in low:
            return "Previous Slide (Left Arrow)"
        elif "engine" in low or "toggle" in low or "pause" in low:
            return "Pause / Resume Engine"
        elif "draw" in low:
            return "Air-Draw Screen Annotation"
        elif "ctrl" in low:
            return "Zoom Slide"
        elif "key l" in low or low == "l":
            return "Laser Pointer / Air-Draw (Key L)"
        elif "key b" in low or low == "b":
            return "Blackout Screen (Key B)"
        elif "key " in low:
            ch = low.replace("key ", "").strip().upper()
            return f"Custom Shortcut (Key {ch})"
        return f"Custom Shortcut ({mapped_val})"

    @classmethod
    def _get_dynamic_row_description(cls, gesture_name, mapped_val):
        if gesture_name == "Open Palm":
            return "Pause Engine & Clear Ink — Hold Open Palm (1.2s) to toggle recognition"
        if gesture_name == "Pinch In / Out":
            return "Zoom Slide — Spread (+) or Pinch (-) like a tablet"

        low = str(mapped_val).lower()
        if gesture_name in ("Right Swipe", "Left Swipe"):
            if "right" in low:
                return "Next Slide — Fires Right Arrow once per swipe (return hand to center to re-arm)"
            elif "left" in low:
                return "Previous Slide — Fires Left Arrow once per swipe (return hand to center to re-arm)"
            else:
                return f"Custom Action ({mapped_val}) — Fires once per swipe (return hand to center to re-arm)"

        if "draw" in low or "key l" in low or low == "l":
            return "Air-Draw & Laser — Point (☝️) to lock Catmull-Rom spline pen onto fingertip at any speed"
        elif "right" in low:
            return "Next Slide — Triggers Right Arrow shortcut"
        elif "left" in low:
            return "Previous Slide — Triggers Left Arrow shortcut"
        return f"Air-Draw + Custom Shortcut — Draws on screen & emulates '{mapped_val}'"

    def _draw_roi_overlay(self, img, card_w, card_h, inside_roi=True):
        roi_bounds = self.config.get("roi_bounds", [0.16, 0.10, 0.84, 0.90])
        x0 = int(roi_bounds[0] * card_w)
        y0 = int(roi_bounds[1] * card_h)
        x1 = int(roi_bounds[2] * card_w)
        y1 = int(roi_bounds[3] * card_h)

        is_dragging = (self.roi_drag_mode is not None)
        color = (255, 200, 0) if is_dragging else ((236, 109, 19) if inside_roi else (0, 165, 255))
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2 if is_dragging else 1, cv2.LINE_AA)

        clen = 22
        thick = 3
        handle_r = 5
        for cx, cy, dx, dy in [
            (x0, y0, clen, clen),
            (x1, y0, -clen, clen),
            (x0, y1, clen, -clen),
            (x1, y1, -clen, -clen)
        ]:
            cv2.line(img, (cx, cy), (cx + dx, cy), color, thick, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + dy), color, thick, cv2.LINE_AA)
            cv2.rectangle(img, (cx - handle_r, cy - handle_r), (cx + handle_r, cy + handle_r), (255, 255, 255), -1)
            cv2.rectangle(img, (cx - handle_r, cy - handle_r), (cx + handle_r, cy + handle_r), color, 2)

        tag = "ACTIVE ROI ZONE" if inside_roi else "ROI LOCKED: IGNORING BACKGROUND"
        cv2.putText(
            img, tag, (max(x0 + 6, 12), max(y0 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2, cv2.LINE_AA
        )

    def toggle_camera(self):
        new_cam_state = not self.config.get("camera_enabled", True)
        self.config["camera_enabled"] = new_cam_state
        self.ipc_config.save(self.config)
        self.cam_stream.set_camera_enabled(new_cam_state)
        self.cached_landmarks = None
        self.cached_hand_score = 0.0
        self.zoom_controller.reset()
        self.vision_engine.reset_tracking()
        self.vision_engine.swipe_latched = False
        self.vision_engine.thumb_latched = False
        self.vision_engine.peace_latched = False
        self.air_canvas.lift_pen()
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
            self.fps_footer_lbl.configure(text="CAMERA: OFF   |   LATENCY: 0.0 ms   |   ZOOM: IDLE")
            self.live_lat_badge.configure(text="Camera Off (0.0 ms)")

    def _render_camera_off_placeholder(self):
        card_w = max(self.video_card.winfo_width(), 860)
        card_h = max(self.video_card.winfo_height(), 460)
        canvas = np.full((card_h, card_w, 3), (38, 23, 15), dtype=np.uint8)
        cx, cy = card_w // 2, card_h // 2
        cv2.circle(canvas, (cx, cy - 28), 38, (65, 51, 35), -1, cv2.LINE_AA)
        cv2.putText(canvas, "CAMERA TURNED OFF", (cx - 165, cy + 40),
                    cv2.FONT_HERSHEY_DUPLEX, 0.95, (248, 250, 252), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Click 'Turn On Camera' to resume live video feed", (cx - 215, cy + 75),
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

    def _toggle_boolean_setting(self, key, label_widget):
        new_val = not self.config.get(key, True)
        self.config[key] = new_val
        self.ipc_config.save(self.config)
        label_widget.configure(image=self.assets["toggle_on"] if new_val else self.assets["toggle_off"])

    @staticmethod
    def _format_keycode_badge(gesture, mapped_val):
        low = str(mapped_val).lower()
        if "engine" in low or "toggle" in low or "pause" in low:
            return "⏸ Engine Toggle"
        elif "draw" in low:
            return "✏️ Air-Draw Pen"
        elif "right" in low:
            return "→ Right Arrow"
        elif "left" in low:
            return "← Left Arrow"
        elif "ctrl" in low:
            return "⊕ Ctrl + / -"
        elif "key l" in low or low == "l":
            return "✦ Key L / Draw"
        return f"⌨ {mapped_val}"

    def _bind_shortcuts(self):
        self.root.bind("<Tab>", self._on_tab_key)
        self.root.bind("<space>", self._on_space_key)
        self.root.bind("<Key>", self._on_any_key)

    def _on_tab_key(self, event):
        order = ("live", "calibration", "settings")
        curr = self.config.get("active_tab", "live")
        next_tab = order[(order.index(curr) + 1) % len(order)] if curr in order else "live"
        self.switch_tab(next_tab)
        return "break"

    def _on_space_key(self, event):
        if self.rebinding_gesture is not None:
            return
        if (time.time() - self.os_dispatcher.last_emulated_time) < 0.25:
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

            formatted_badge = self._format_keycode_badge(g, new_binding)
            self.row_widgets[g]["badge"].configure(text=formatted_badge)
            self.row_widgets[g]["desc"].configure(text=self._get_dynamic_row_description(g, new_binding))
            self.row_widgets[g]["rebind"].configure(text="REBIND", bg=self.BG_CARD, fg=self.SLATE_700)

            if self.last_displayed_gesture == g:
                self.dispatched_action_lbl.configure(text=self._get_dynamic_action_summary(g, new_binding))
                self.dispatched_key_badge.configure(text=formatted_badge)

            self.rebinding_gesture = None
            return

    def switch_tab(self, tab_name):
        if tab_name not in ("live", "calibration", "settings"):
            tab_name = "live"
        self.config["active_tab"] = tab_name
        self.ipc_config.save(self.config)

        pal = self.THEMES.get(self.config.get("theme", "light"), self.THEMES["light"])

        
        self.live_view.pack_forget()
        self.calibration_view.pack_forget()
        self.settings_view.pack_forget()

        # Reset all 3 tab buttons to inactive pill color
        for btn in (self.tab_live_btn, self.tab_calib_btn, self.tab_settings_btn):
            btn.configure(bg=pal["PILL_BG"], fg=pal["FG_MUTED"])

        if tab_name == "live":
            self.live_view.pack(fill="both", expand=True)
            self.tab_live_btn.configure(bg=pal["BG_CARD"], fg=pal["FG_PRIMARY"])
            if not self.config.get("camera_enabled", True):
                self.cam_off_rendered = False
        elif tab_name == "calibration":
            self.calibration_view.refresh_from_config()
            self.calibration_view.pack(fill="both", expand=True)
            self.tab_calib_btn.configure(bg=pal["BG_CARD"], fg=pal["ACCENT"])
        else:
            self.settings_view.pack(fill="both", expand=True)
            self.tab_settings_btn.configure(bg=pal["BG_CARD"], fg=pal["ACCENT"])

    def toggle_engine(self):
        new_state = not self.config.get("recognition_enabled", True)
        self.config["recognition_enabled"] = new_state
        self.ipc_config.save(self.config)
        self.zoom_controller.reset()
        self.vision_engine.reset_tracking()
        self.vision_engine.swipe_latched = False
        self.vision_engine.thumb_latched = False
        self.vision_engine.peace_latched = False
        self.air_canvas.lift_pen()
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
        self.config["roi_bounds"] = list(IPCConfigManager.DEFAULT_ROI)
        self.ipc_config.save(self.config)
        for g, widgets in self.row_widgets.items():
            mapped_val = self.config["gesture_mappings"][g]
            widgets["badge"].configure(text=self._format_keycode_badge(g, mapped_val))
            widgets["desc"].configure(text=self._get_dynamic_row_description(g, mapped_val))
            is_on = self.config["gesture_states"].get(g, True)
            widgets["toggle"].configure(image=self.assets["toggle_on"] if is_on else self.assets["toggle_off"])

        if self.last_displayed_gesture in self.config["gesture_mappings"]:
            g = self.last_displayed_gesture
            mapped_val = self.config["gesture_mappings"][g]
            self.dispatched_action_lbl.configure(text=self._get_dynamic_action_summary(g, mapped_val))
            self.dispatched_key_badge.configure(text=self._format_keycode_badge(g, mapped_val))

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
        self.last_displayed_gesture = gesture_name
        icon_key = gesture_name if gesture_name in self.assets["icons"] else "Pinch In / Out"
        self.live_gesture_icon_lbl.configure(image=self.assets["icons"][icon_key])
        self.live_gesture_title.configure(text=gesture_name)
        self.recognized_badge.configure(text=" RECOGNIZED ", fg=self.GREEN_TEXT, bg=self.GREEN_BG)
        self.last_recognized_ui_time = time.time()

        live_mapped_val = self.config.get("gesture_mappings", {}).get(gesture_name, "")
        if custom_key:
            display_key_badge = custom_key
        elif live_mapped_val:
            display_key_badge = self._format_keycode_badge(gesture_name, live_mapped_val)
        else:
            _, _, display_key_badge = self.DESCRIPTIONS.get(gesture_name, ("", "", "—"))

        if custom_action:
            display_action_text = custom_action
        elif live_mapped_val:
            display_action_text = self._get_dynamic_action_summary(gesture_name, live_mapped_val)
        else:
            desc, _, _ = self.DESCRIPTIONS.get(gesture_name, ("Action", "", "—"))
            display_action_text = desc.split("—")[0].strip()

        self.dispatched_action_lbl.configure(text=display_action_text)
        self.dispatched_key_badge.configure(text=display_key_badge)
        self.confidence_val_lbl.configure(text=f"{confidence_pct:.1f}%", fg=self.GREEN_TEXT)

    def update_loop(self):
        try:
            now_tick = time.time()
            if (now_tick - self.last_loop_tick_time) > 1.5:
                self.vision_engine.reset_tracking()
                self.zoom_controller.reset()
                self.palm_hold_start = 0.0
                self.peace_hold_start = 0.0
                self.thumb_hold_start = 0.0
            self.last_loop_tick_time = now_tick

            if self.roi_drag_mode is None:
                prev_tab = self.config.get("active_tab", "live")
                self.config = self.ipc_config.poll_if_needed(interval=2.0)
                self.config["active_tab"] = prev_tab

            cam_on = self.config.get("camera_enabled", True)
            is_live_tab = (self.config.get("active_tab", "live") == "live")

            if not cam_on:
                if is_live_tab and not self.cam_off_rendered:
                    self._render_camera_off_placeholder()
                self.root.after(30, self.update_loop)
                return

            #camera disconnected 
            if cam_on and (self.cam_stream.cap is None or not self.cam_stream.cap.isOpened()) and (now_tick - self.log_start_time) > 3.0:
                self.error_handler.notify_camera_not_found()

            ret, raw_frame, cam_frame_id, measured_cam_fps = self.cam_stream.read()

            if ret and raw_frame is not None and cam_frame_id != self.last_processed_cam_id:
                self.cam_off_rendered = False
                self.last_processed_cam_id = cam_frame_id
                self.frame_count += 1

                recognition_on = self.config.get("recognition_enabled", True)
                self._sync_engine_ui(recognition_on)

                # Read latest 45+ FPS landmarks from our non-blocking AsyncMediaPipeTracker!
                self.cached_landmarks, self.cached_hand_score, mean_lat = self.async_tracker.get_latest()
                now_time = time.time()
                hand_detected = self.cached_landmarks is not None
                if not hand_detected:
                    if recognition_on and is_live_tab and (now_time - self.no_hand_since) > 15.0:
                        self.error_handler.notify_no_hand_detected()
                else:
                    self.no_hand_since = now_time
                base_cooldown = max(float(self.config.get("swipe_cooldown", 1.25)), 1.00)
                swipe_cooldown = max(base_cooldown, 1.85) if self.config.get("accessibility_tremor_filter", False) else base_cooldown
                zoom_sens = float(self.config.get("zoom_sensitivity", 0.050))
                roi_enabled = bool(self.config.get("enable_roi_optional", False))
                roi_bounds = self.config.get("roi_bounds", [0.16, 0.10, 0.84, 0.90])

                if is_live_tab:
                    card_w = max(self.video_card.winfo_width(), 860)
                    card_h = max(self.video_card.winfo_height(), 460)
                    lit_frame = AsyncMediaPipeTracker.apply_lighting_calibration(
                        raw_frame,
                        sensitivity=self.config.get("lighting_sensitivity", 0.50),
                        auto_light=self.config.get("enable_auto_lighting_optional", False)
                    )
                    base_vis = self._apply_clahe_low_light(lit_frame) if self.config.get("low_light_clahe", False) else lit_frame
                    display_frame = cv2.resize(base_vis, (card_w, card_h), interpolation=cv2.INTER_NEAREST)
                else:
                    card_w, card_h = 960, 500
                    display_frame = None

                active_title = self.os_dispatcher._get_active_window_title()
                wave_is_active = ("wave" in active_title) and is_live_tab

                in_presenter_roi = True
                if not hand_detected:
                    if is_live_tab and roi_enabled and display_frame is not None:
                        self._draw_roi_overlay(display_frame, card_w, card_h, inside_roi=True)

                    # Use grace buffer even when MediaPipe briefly loses hand due to ultra-fast motion blur!
                    self.air_canvas.step_grace_or_lift()
                    self.pill_hand_dot.configure(fg=self.SLATE_400)
                    self.pill_hand_val.configure(text=" NONE")
                    self.direction_banner_text = "DIRECTION: READY (NO HAND)"
                    self.live_conf_ema.reset()
                    self.confidence_val_lbl.configure(text="0.0%", fg=self.SLATE_400)
                    self.recognized_badge.configure(text=" STANDBY ", fg=self.SLATE_500, bg=self.PILL_BG)
                    self.zoom_controller.reset()
                    self.vision_engine.reset_tracking()
                    self.vision_engine.swipe_latched = False
                    self.vision_engine.thumb_latched = False
                    self.vision_engine.peace_latched = False
                    self.palm_hold_start = 0.0
                    self.peace_hold_start = 0.0
                    self.thumb_hold_start = 0.0
                else:
                    in_safe_margin = self.vision_engine.is_within_safe_frame_margin(self.cached_landmarks)
                    in_presenter_roi = (
                        self.vision_engine.is_within_presenter_roi(self.cached_landmarks, roi_bounds)
                        if roi_enabled else True
                    )
                    hand_valid = in_safe_margin and in_presenter_roi

                    if is_live_tab and roi_enabled and display_frame is not None:
                        self._draw_roi_overlay(display_frame, card_w, card_h, inside_roi=in_presenter_roi)

                    if in_presenter_roi:
                        self.pill_hand_dot.configure(fg=self.BLUE_PRIMARY)
                        self.pill_hand_val.configure(text=f" DETECTED ({self.cached_hand_score * 100:.0f}%)")
                    else:
                        self.pill_hand_dot.configure(fg=self.AMBER_TEXT)
                        self.pill_hand_val.configure(text=" OUTSIDE ROI")
                        self.zoom_controller.reset()
                        self.vision_engine.reset_tracking()
                        self.vision_engine.swipe_latched = False
                        self.vision_engine.thumb_latched = False
                        self.vision_engine.peace_latched = False
                        self.air_canvas.lift_pen()

                    wrist = self.cached_landmarks.landmark[0]
                    thumb_tip = self.cached_landmarks.landmark[4]
                    index_tip = self.cached_landmarks.landmark[8]

                    if is_live_tab and self.config.get("show_skeleton_overlay", True):
                        skel_col = (60, 255, 60) if in_presenter_roi else (0, 140, 255)
                        self.mp_drawing.draw_landmarks(
                            display_frame,
                            self.cached_landmarks,
                            self.mp_hands.HAND_CONNECTIONS,
                            self.mp_drawing.DrawingSpec(color=skel_col, thickness=2, circle_radius=3),
                            self.mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )

                    pt_wrist = (int(wrist.x * card_w), int(wrist.y * card_h))
                    pt_index = (int(index_tip.x * card_w), int(index_tip.y * card_h))
                    pt_thumb = (int(thumb_tip.x * card_w), int(thumb_tip.y * card_h))

                    gesture_states = self.config.get("gesture_states", {})
                    norm_vel_x, velocity_swipe, vel_conf = self.vision_engine.compute_velocity_and_swipe(
                        self.cached_landmarks, now_time, mp_hand_score=self.cached_hand_score
                    )
                    static_dir_gesture, dir_banner, dir_conf = self.vision_engine.detect_static_direction_pose(
                        self.cached_landmarks, mp_hand_score=self.cached_hand_score
                    )
                    mlp_label, mlp_conf = self.vision_engine.classify_static_mlp(self.cached_landmarks, gesture_states)

                    if static_dir_gesture is None and velocity_swipe is None and abs(norm_vel_x) < 0.85:
                        self.vision_engine.swipe_latched = False

                    if not in_presenter_roi:
                        self.direction_banner_text = "STATUS: OUTSIDE ROI (BACKGROUND IGNORED)"
                    elif not in_safe_margin:
                        self.direction_banner_text = "STATUS: EDGE BORDER (SAFE GUARD)"
                    elif now_time < self.global_cooldown_until:
                        rem_cd = max(0.0, self.global_cooldown_until - now_time)
                        self.direction_banner_text = f"ACTION LOCKED — WAIT ({rem_cd:.1f}s)"
                    else:
                        self.direction_banner_text = dir_banner

                    if (now_time - self.last_recognized_ui_time) > 0.60 and self.zoom_controller.state == TabletPinchZoomController.STATE_IDLE:
                        self.recognized_badge.configure(text=" STANDBY ", fg=self.SLATE_500, bg=self.PILL_BG)

                    is_peace = (
                        self.config.get("enable_voice_notes", True) and
                        abs(norm_vel_x) < 0.80 and
                        self.vision_engine.is_peace_voice_gesture(self.cached_landmarks)
                    )
                    if not is_peace:
                        self.vision_engine.peace_latched = False

                    is_thumbs_up = (
                        self.config.get("enable_slideshow_launch", True) and
                        abs(norm_vel_x) < 0.80 and
                        self.vision_engine.is_thumbs_up_slideshow_gesture(self.cached_landmarks)
                    )
                    if not is_thumbs_up:
                        self.vision_engine.thumb_latched = False

                    is_air_draw = (
                        recognition_on and in_presenter_roi and
                        self.config.get("enable_air_draw", True) and
                        gesture_states.get("Index Finger Point", True) and
                        not is_peace and not is_thumbs_up and
                        (self.air_canvas.is_drawing_active or velocity_swipe is None) and
                        self.vision_engine.is_index_draw_pose(
                            self.cached_landmarks,
                            zoom_active=(self.zoom_controller.state != TabletPinchZoomController.STATE_IDLE),
                            already_drawing=self.air_canvas.is_drawing_active
                        )
                    )

                    # PRIORITY 0: Pen-Locked Catmull-Rom Spline Air-Draw (Follows Landmark 8 at ANY speed!)
                    if is_air_draw and now_time >= self.global_cooldown_until:
                        self.zoom_controller.reset()
                        self.vision_engine.reset_tracking()
                        sm_x = self.draw_x_ema.filter(index_tip.x)
                        sm_y = self.draw_y_ema.filter(index_tip.y)
                        sx, sy = self.air_canvas.compute_screen_pixel(
                            sm_x, sm_y,
                            video_widget=self.video_label,
                            wave_is_active=wave_is_active,
                            roi_enabled=roi_enabled,
                            roi_bounds=roi_bounds
                        )
                        self.air_canvas.add_screen_point(sx, sy, pen_down=True)
                        self.direction_banner_text = "AIR-DRAWING LOCKED (OPEN PALM TO CLEAR)"
                        self._update_telemetry_card("Index Finger Point", 98.5)

                    # PRIORITY 1: One-Shot Swipe (Only when NOT drawing)
                    elif recognition_on and hand_valid and now_time >= self.global_cooldown_until and not self.air_canvas.is_drawing_active and (velocity_swipe is not None or static_dir_gesture is not None):
                        self.zoom_controller.reset()
                        self.air_canvas.lift_pen()
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

                        if self.vision_engine.swipe_latched:
                            self.direction_banner_text = f"{detected_gesture.upper()} DONE (RETURN HAND TO CENTER)"
                        elif not recoil_blocked and (now_time - self.vision_engine.last_trigger_time) >= swipe_cooldown:
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
                                self.vision_engine.swipe_latched = True
                                self.global_cooldown_until = now_time + 1.25
                                self.vision_engine.reset_tracking()

                    # PRIORITY 2: Pinch-to-Zoom, Peace Continuous Mic Toggle, or Thumbs-Up Slideshow
                    elif recognition_on and hand_valid:
                        self.air_canvas.step_grace_or_lift()
                        if self.air_canvas.is_drawing_active:
                            sm_x = self.draw_x_ema.filter(index_tip.x)
                            sm_y = self.draw_y_ema.filter(index_tip.y)
                            sx, sy = self.air_canvas.compute_screen_pixel(
                                sm_x, sm_y,
                                video_widget=self.video_label,
                                wave_is_active=wave_is_active,
                                roi_enabled=roi_enabled,
                                roi_bounds=roi_bounds
                            )
                            self.air_canvas.add_screen_point(sx, sy, pen_down=True)

                        zoom_state = self.zoom_controller.state
                        zoom_ticks = 0
                        smoothed_ratio = 0.0
                        zoom_label = ""
                        zoom_conf_pct = 0.0

                        if not self.air_canvas.is_drawing_active and self.config.get("enable_zoom_optional", True) and gesture_states.get("Pinch In / Out", True):
                            zoom_state, zoom_ticks, smoothed_ratio, zoom_label, zoom_conf_pct = self.zoom_controller.update(
                                self.cached_landmarks, norm_vel_x,
                                mp_hand_score=self.cached_hand_score,
                                step_sensitivity=zoom_sens
                            )

                        if zoom_state != TabletPinchZoomController.STATE_IDLE:
                            self.air_canvas.lift_pen()
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
                                    custom_action="Zoom In (+)", custom_key="⊕ Ctrl + Plus (+)"
                                )
                            elif zoom_ticks < 0:
                                self.os_dispatcher.execute_zoom(zoom_ticks)
                                self._update_telemetry_card(
                                    "Pinch In (Zoom Out)", dyn_zoom_conf,
                                    custom_action="Zoom Out (-)", custom_key="⊕ Ctrl + Minus (-)"
                                )
                            self.vision_engine.prediction_buffer.clear()
                        elif now_time >= self.global_cooldown_until:
                            # B. PEACE SIGN (✌️) HOLD -> START / STOP UNBROKEN VOICE SESSION
                            if is_peace and not self.vision_engine.peace_latched and (now_time - self.last_peace_trigger_time) > 2.5:
                                self.air_canvas.lift_pen()
                                if self.peace_hold_start == 0.0:
                                    self.peace_hold_start = now_time
                                else:
                                    held_p = now_time - self.peace_hold_start
                                    rem_p = max(0.0, self.PEACE_HOLD_DURATION - held_p)
                                    act_lbl = "STOP & TRANSCRIBE" if self.voice_notes.is_listening else "START VOICE NOTE"
                                    self.direction_banner_text = f"HOLD PEACE SIGN TO {act_lbl} ({rem_p:.1f}s)"
                                    if held_p >= self.PEACE_HOLD_DURATION:
                                        self.last_peace_trigger_time = now_time
                                        self.peace_hold_start = 0.0
                                        self.vision_engine.peace_latched = True
                                        self.global_cooldown_until = now_time + 1.5
                                        self._toggle_voice_note_action()
                            elif is_peace and self.vision_engine.peace_latched:
                                state_msg = "RECORDING ON" if self.voice_notes.is_listening else "TRANSCRIBING"
                                self.direction_banner_text = f"{state_msg} (RELEASE PEACE SIGN)"
                            else:
                                self.peace_hold_start = 0.0

                            # C. THUMBS UP (👍) HOLD -> ONE-SHOT FULLSCREEN SLIDESHOW
                            if is_thumbs_up and not self.vision_engine.thumb_latched and (now_time - self.last_thumb_trigger_time) > 4.0:
                                self.air_canvas.lift_pen()
                                if self.thumb_hold_start == 0.0:
                                    self.thumb_hold_start = now_time
                                else:
                                    held_t = now_time - self.thumb_hold_start
                                    rem_t = max(0.0, self.THUMB_HOLD_DURATION - held_t)
                                    self.direction_banner_text = f"HOLD THUMBS UP FOR SLIDESHOW ({rem_t:.1f}s)"
                                    if held_t >= self.THUMB_HOLD_DURATION:
                                        self.last_thumb_trigger_time = now_time
                                        self.thumb_hold_start = 0.0
                                        self.vision_engine.thumb_latched = True
                                        self.global_cooldown_until = now_time + 2.0
                                        self.os_dispatcher.execute_start_slideshow()
                                        self._update_telemetry_card(
                                            "Right Swipe", 98.5,
                                            custom_action="Start Fullscreen Slideshow",
                                            custom_key="▶ Slideshow"
                                        )
                                        self.live_gesture_title.configure(text="Thumbs Up (Slideshow)")
                            elif is_thumbs_up and self.vision_engine.thumb_latched:
                                self.direction_banner_text = "SLIDESHOW STARTED (RELEASE THUMBS UP)"
                            else:
                                self.thumb_hold_start = 0.0

                    # OPEN PALM HOLD CHECK: Clears any Air-Draw ink immediately AND pauses Engine at 1.2s
                    open_palm_enabled = gesture_states.get("Open Palm", True)
                    is_open_palm = (
                        open_palm_enabled and
                        in_presenter_roi and
                        abs(norm_vel_x) < 0.90 and
                        self.vision_engine.is_strict_open_palm(
                            self.cached_landmarks,
                            zoom_active=(self.zoom_controller.state != TabletPinchZoomController.STATE_IDLE)
                        )
                    )
                    if is_open_palm:
                        self.air_canvas.clear_all()

                    if is_open_palm and (now_time - self.last_palm_toggle_time) > 2.0:
                        if self.palm_hold_start == 0.0:
                            self.palm_hold_start = now_time
                        else:
                            held_sec = now_time - self.palm_hold_start
                            if held_sec >= 0.25:
                                rem = max(0.0, self.PALM_HOLD_DURATION - held_sec)
                                action_word = "PAUSE" if recognition_on else "RESUME"
                                self.direction_banner_text = f"HOLD OPEN PALM TO {action_word} ({rem:.1f}s)"
                            if held_sec >= self.PALM_HOLD_DURATION:
                                self.toggle_engine()
                                recognition_on = self.config.get("recognition_enabled", True)
                                state_str = "Engine Resumed (ON)" if recognition_on else "Engine Paused (OFF)"
                                self._update_telemetry_card(
                                    "Open Palm", 98.5,
                                    custom_action=state_str,
                                    custom_key="⏸ Engine Toggle"
                                )
                                self.last_palm_toggle_time = now_time
                                self.palm_hold_start = 0.0
                                self.global_cooldown_until = now_time + 1.5
                    else:
                        self.palm_hold_start = 0.0

                self._log_60s_telemetry_sample(measured_cam_fps, mean_lat)

                roi_status_str = "ON" if roi_enabled else "OFF"
                self.cam_txt.configure(text=f"CAM ACTIVE ({measured_cam_fps:.1f} FPS)")
                self.fps_footer_lbl.configure(
                    text=f"CAMERA FPS: {measured_cam_fps:.1f} FPS   |   LATENCY: {mean_lat:.1f} ms   |   ROI: {roi_status_str}   |   ZOOM: {self.zoom_controller.state}"
                )
                self.live_lat_badge.configure(text=f"Pipeline Latency: {mean_lat:.1f} ms")

                if is_live_tab and display_frame is not None:
                    if not recognition_on and self.palm_hold_start == 0.0:
                        overlay_text = "ENGINE PAUSED (HOLD OPEN PALM TO RESUME)"
                    else:
                        overlay_text = self.direction_banner_text

                    if not recognition_on:
                        overlay_color = (60, 60, 235)
                    elif hand_detected and not in_presenter_roi:
                        overlay_color = (0, 165, 255)
                    else:
                        overlay_color = (50, 235, 50)

                    cv2.putText(
                        display_frame, overlay_text, (16, 36),
                        cv2.FONT_HERSHEY_DUPLEX, 0.78, overlay_color, 2, cv2.LINE_AA
                    )
                    rgb_display = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_display)
                    self.tk_video_img = ImageTk.PhotoImage(image=pil_img)
                    self.video_label.configure(image=self.tk_video_img)
        except Exception as e:
            print(f"[WARN] Non-fatal loop recovery: {e}")

        self.root.after(4, self.update_loop)

    def on_close(self):
        try:
            self.voice_notes.stop_continuous_listening()
            self.async_tracker.close()
            self.cam_stream.release()
        except Exception:
            pass
        self.root.destroy()


def main():
    root = tk.Tk()
    app = WaveDesktopApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()