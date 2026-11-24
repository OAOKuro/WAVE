# error_handler.py
import time
import tkinter as tk
from tkinter import messagebox


class ErrorHandler:
    """
    Actionable Error Handling Module (FYP Section 4.1.1 & 4.2.4).
    Supports both modal dialogs and non-blocking in-app actionable banners
    with cooldown throttling so real-time video tracking never freezes.
    """

    def __init__(self, root=None, callbacks=None):
        self.root = root
        # callbacks: dict with optional keys:
        # "retry_camera", "open_calibration", "reset_settings"
        self.callbacks = callbacks or {}
        self._last_shown = {}
        self._active_banner = None

    def _can_notify(self, error_key, cooldown=12.0):
        now = time.time()
        if (now - self._last_shown.get(error_key, 0.0)) < cooldown:
            return False
        self._last_shown[error_key] = now
        return True

    def dismiss_banner(self):
        if self._active_banner is not None:
            try:
                self._active_banner.destroy()
            except Exception:
                pass
            self._active_banner = None

    def _show_actionable_popup(self, error_key, icon, title, message, steps, action_label, action_cb, cooldown=12.0):
        if self.root is None:
            messagebox.showwarning(title, f"{message}\n\nSuggested Actions:\n" + "\n".join(steps))
            return

        if not self._can_notify(error_key, cooldown=cooldown):
            return

        self.dismiss_banner()

        popup = tk.Toplevel(self.root)
        popup.title(title)
        popup.overrideredirect(True)
        popup.attributes("-topmost", True)
        popup.configure(bg="#1E293B", highlightthickness=2, highlightbackground="#EF4444")

        # Position cleanly at bottom-right of the main window
        self.root.update_idletasks()
        rx = self.root.winfo_rootx() + max(self.root.winfo_width() - 440, 20)
        ry = self.root.winfo_rooty() + max(self.root.winfo_height() - 250, 40)
        popup.geometry(f"415x220+{rx}+{ry}")
        self._active_banner = popup

        # Header
        head = tk.Frame(popup, bg="#1E293B", padx=16, pady=10)
        head.pack(fill="x")
        tk.Label(head, text=f"{icon}  {title}", font=("Segoe UI", 12, "bold"),
                 fg="#F87171", bg="#1E293B").pack(side="left")

        # Plain-language description
        tk.Label(popup, text=message, font=("Segoe UI", 9),
                 fg="#F8FAFC", bg="#1E293B", anchor="w", justify="left", wraplength=380).pack(fill="x", padx=16)

        # Suggested Actions Box
        act_box = tk.Frame(popup, bg="#0F172A", padx=12, pady=8, highlightthickness=1, highlightbackground="#334155")
        act_box.pack(fill="x", padx=16, pady=10)
        tk.Label(act_box, text="Suggested Actions", font=("Segoe UI", 8, "bold"),
                 fg="#34D399", bg="#0F172A").pack(anchor="w")
        for idx, step in enumerate(steps, 1):
            tk.Label(act_box, text=f"{idx}. {step}", font=("Segoe UI", 8),
                     fg="#CBD5E1", bg="#0F172A").pack(anchor="w")

        # Action Buttons
        btn_row = tk.Frame(popup, bg="#1E293B", padx=16, pady=6)
        btn_row.pack(fill="x", side="bottom")

        def _on_action():
            self.dismiss_banner()
            if callable(action_cb):
                action_cb()

        if action_label:
            act_btn = tk.Label(
                btn_row, text=f"⟳ {action_label}", font=("Segoe UI", 9, "bold"),
                fg="#0F172A", bg="#10B981", padx=12, pady=5, cursor="hand2"
            )
            act_btn.pack(side="right", padx=(8, 0))
            act_btn.bind("<Button-1>", lambda e: _on_action())

        dis_btn = tk.Label(
            btn_row, text="✕ Dismiss", font=("Segoe UI", 9, "bold"),
            fg="#F8FAFC", bg="#475569", padx=12, pady=5, cursor="hand2"
        )
        dis_btn.pack(side="right")
        dis_btn.bind("<Button-1>", lambda e: self.dismiss_banner())

        # Auto-dismiss after 9 seconds so it never blocks a live presentation
        popup.after(9000, self.dismiss_banner)

    def notify_camera_not_found(self):
        self._show_actionable_popup(
            error_key="camera_not_found",
            icon="📷",
            title="Camera Not Found",
            message="No webcam detected. Please connect or enable a camera.",
            steps=[
                "Check physical USB or built-in camera connections",
                "Ensure no other app (Zoom/Teams) is locking the camera",
                "Click 'Check Camera' to re-initialize sensor"
            ],
            action_label="Check Camera",
            action_cb=self.callbacks.get("retry_camera"),
            cooldown=15.0
        )

    def notify_no_hand_detected(self):
        self._show_actionable_popup(
            error_key="no_hand_detected",
            icon="✋",
            title="No Hand Detected",
            message="No hand detected in camera view for an extended period.",
            steps=[
                "Raise your hand inside the camera frame / ROI box",
                "Ensure adequate front lighting on your palm",
                "Adjust camera angle or open Calibration"
            ],
            action_label="Open Calibration",
            action_cb=self.callbacks.get("open_calibration"),
            cooldown=25.0
        )

    def notify_low_confidence(self):
        self._show_actionable_popup(
            error_key="low_confidence",
            icon="⚠️",
            title="Low Confidence",
            message="Gesture not recognized clearly due to lighting or threshold.",
            steps=[
                "Adjust Lighting Sensitivity in the Calibration tab",
                "Keep your hand steady and within 0.5m–1.5m of webcam",
                "Lower Minimum Detection Confidence if needed"
            ],
            action_label="Open Calibration",
            action_cb=self.callbacks.get("open_calibration"),
            cooldown=18.0
        )

    def notify_config_error(self):
        self._show_actionable_popup(
            error_key="config_error",
            icon="💾",
            title="Configuration Error",
            message="Settings file (config.json) was corrupted or had invalid values.",
            steps=[
                "Default safe settings have been loaded automatically",
                "Click 'Reset Settings' to overwrite config.json cleanly"
            ],
            action_label="Reset Settings",
            action_cb=self.callbacks.get("reset_settings"),
            cooldown=5.0
        )

    # Static fallbacks preserved for backward compatibility
    @staticmethod
    def camera_not_found():
        return messagebox.showerror(
            "Camera Not Found",
            "No webcam detected. Please connect a camera.\n\n"
            "Suggested Actions:\n1. Check camera connections\n2. Ensure no other app is using the camera"
        )

    @staticmethod
    def no_hand_detected():
        return messagebox.showwarning(
            "No Hand Detected",
            "No hand detected in camera view.\n\n"
            "Suggested Actions:\n1. Show your hand to the camera\n2. Ensure adequate lighting"
        )

    @staticmethod
    def low_confidence():
        return messagebox.showwarning(
            "Low Confidence",
            "Gesture not recognized clearly.\n\n"
            "Suggested Actions:\n1. Adjust lighting sensitivity in Calibration tab\n2. Lower confidence threshold"
        )

    @staticmethod
    def config_error():
        return messagebox.showwarning(
            "Configuration Error",
            "Settings file could not be loaded.\n\n"
            "Suggested Actions:\n1. Check config.json exists\n2. Reset to default settings"
        )