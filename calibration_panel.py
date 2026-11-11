# calibration_panel.py (Tkinter version)
import tkinter as tk
from tkinter import messagebox

class CalibrationPanel(tk.Frame):
    """
    Calibration tab that plugs into WaveDesktopApp.
    Reads/writes the same config.json keys as the partner's IPCConfigManager.
    """

    def __init__(self, parent, ipc_config, on_save_callback=None):
        super().__init__(parent, bg="#F8FAFC")
        self.ipc_config = ipc_config
        self.config = ipc_config.state
        self.on_save_callback = on_save_callback
        self._build_ui()

    def _build_ui(self):
        # Title
        tk.Label(
            self, text="Calibration Settings",
            font=("Segoe UI", 18, "bold"), fg="#0F172A", bg="#F8FAFC"
        ).pack(anchor="w", padx=40, pady=(30, 4))

        tk.Label(
            self, text="Adjust detection parameters to match your environment",
            font=("Segoe UI", 10), fg="#64748B", bg="#F8FAFC"
        ).pack(anchor="w", padx=40, pady=(0, 20))

        # Slider 1: Lighting Sensitivity
        self.lighting_slider = self._build_slider(
            "Lighting Sensitivity",
            "Adjusts frame brightness before detection (0.0 = darkest, 1.0 = brightest)",
            "lighting_sensitivity", 0.0, 1.0, 0.6, step=0.05
        )

        # Slider 2: Minimum Detection Confidence
        self.conf_slider = self._build_slider(
            "Minimum Detection Confidence",
            "Lower = detects more gestures (may increase false positives). Floor is 0.7 in main.py",
            "min_detection_confidence", 0.5, 1.0, 0.7, step=0.05
        )

        # Checkbox: ROI
        self.roi_var = self._build_checkbox(
            "Enable Region of Interest (ROI)",
            "Focus hand detection on a specific area of the camera frame",
            "enable_roi_optional", False
        )

        # Checkbox: Auto Lighting
        self.auto_light_var = self._build_checkbox(
            "Enable Auto Lighting Adjustment",
            "Automatically adjust brightness based on ambient light",
            "enable_auto_lighting_optional", False
        )

        # Save Button
        tk.Button(
            self, text="Save Calibration", font=("Segoe UI", 10, "bold"),
            bg="#136DEC", fg="white", padx=20, pady=8, bd=0, cursor="hand2",
            command=self._save_calibration
        ).pack(anchor="w", padx=40, pady=25)

    def _build_slider(self, title, description, key, min_val, max_val, default, step):
        card = tk.Frame(self, bg="#FFFFFF", highlightthickness=1,
                        highlightbackground="#E2E8F0", padx=20, pady=15)
        card.pack(fill="x", padx=40, pady=6)

        tk.Label(card, text=title, font=("Segoe UI", 11, "bold"),
                 fg="#0F172A", bg="#FFFFFF").pack(anchor="w")
        tk.Label(card, text=description, font=("Segoe UI", 9),
                 fg="#64748B", bg="#FFFFFF", wraplength=700,
                 justify="left").pack(anchor="w", pady=(2, 8))

        value = float(self.config.get(key, default))
        val_label = tk.Label(card, text=f"Current: {value:.2f}",
                             font=("Segoe UI", 10, "bold"), fg="#136DEC", bg="#FFFFFF")
        val_label.pack(anchor="w")

        scale = tk.Scale(
            card, from_=min_val, to=max_val, resolution=step,
            orient="horizontal", length=600, bg="#FFFFFF",
            highlightthickness=0, troughcolor="#E2E8F0", fg="#136DEC",
            showvalue=False,
            command=lambda v, lbl=val_label: lbl.configure(text=f"Current: {float(v):.2f}")
        )
        scale.set(value)
        scale.pack(anchor="w", pady=4)
        return scale

    def _build_checkbox(self, title, description, key, default):
        card = tk.Frame(self, bg="#FFFFFF", highlightthickness=1,
                        highlightbackground="#E2E8F0", padx=20, pady=15)
        card.pack(fill="x", padx=40, pady=6)

        tk.Label(card, text=title, font=("Segoe UI", 11, "bold"),
                 fg="#0F172A", bg="#FFFFFF").pack(anchor="w")
        tk.Label(card, text=description, font=("Segoe UI", 9),
                 fg="#64748B", bg="#FFFFFF").pack(anchor="w", pady=(2, 8))

        var = tk.BooleanVar(value=self.config.get(key, default))
        tk.Checkbutton(card, text="Enabled", variable=var,
                       font=("Segoe UI", 9), bg="#FFFFFF",
                       activebackground="#FFFFFF").pack(anchor="w")
        return var

    def _save_calibration(self):
        """Write calibration values to config.json via the partner's IPCConfigManager."""
        self.config["lighting_sensitivity"] = round(self.lighting_slider.get(), 2)
        self.config["min_detection_confidence"] = round(self.conf_slider.get(), 2)
        self.config["enable_roi_optional"] = self.roi_var.get()
        self.config["enable_auto_lighting_optional"] = self.auto_light_var.get()
        self.ipc_config.save(self.config)

        if self.on_save_callback:
            self.on_save_callback()

        messagebox.showinfo("Calibration", "Calibration saved successfully.")