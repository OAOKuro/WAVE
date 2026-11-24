# calibration_panel.py
import json
import tkinter as tk
from tkinter import filedialog, messagebox


class CalibrationPanel(tk.Frame):
    """
    Calibration & Settings Persistence Tab (FYP Section 4.2.2 & 4.2.5).
    Provides real-time Lighting Sensitivity, Confidence Threshold, Auto-Lighting,
    ROI Coordinates, and Import/Export JSON persistence.
    """

    def __init__(
            self, parent, ipc_config, on_change_callback=None, on_save_callback=None, on_reset_roi=None, sample_provider=None):
        super().__init__(parent)
        self.ipc_config = ipc_config
        self.config = ipc_config.state
        self.on_change_callback = on_change_callback
        self.on_save_callback = on_save_callback
        self.on_reset_roi = on_reset_roi
        self.sample_provider = sample_provider

        self.theme = {}
        self.cards = []
        self._build_ui()

    def _build_ui(self):
        self.wrapper = tk.Frame(self)
        self.wrapper.pack(fill="both", expand=True, padx=56, pady=14)

        # Header Row
        self.header = tk.Frame(self.wrapper)
        self.header.pack(fill="x", pady=(0, 10))

        title_box = tk.Frame(self.header)
        title_box.pack(side="left")

        self.kicker_lbl = tk.Label(
            title_box, text="🎛  ENVIRONMENT & DETECTION CALIBRATION",
            font=("Segoe UI", 8, "bold")
        )
        self.kicker_lbl.pack(anchor="w")

        self.title_lbl = tk.Label(
            title_box, text="Calibration & Persistence Panel",
            font=("Segoe UI", 18, "bold")
        )
        self.title_lbl.pack(anchor="w")

        self.sub_lbl = tk.Label(
            title_box, text="Adjust detection parameters in real time to match room lighting and presenter distance.",
            font=("Segoe UI", 9)
        )
        self.sub_lbl.pack(anchor="w")

        # Top-right Action Buttons (Import / Export / Reset / Save)
        btn_box = tk.Frame(self.header)
        btn_box.pack(side="right", anchor="s")

        self.save_btn = tk.Label(
            btn_box, text="💾 Save Calibration", font=("Segoe UI", 9, "bold"),
            padx=14, pady=7, cursor="hand2"
        )
        self.save_btn.pack(side="right", padx=(8, 0))
        self.save_btn.bind("<Button-1>", lambda e: self._save_calibration(show_popup=True))

        self.reset_btn = tk.Label(
            btn_box, text="↺ Reset Defaults", font=("Segoe UI", 9, "bold"),
            padx=12, pady=7, cursor="hand2"
        )
        self.reset_btn.pack(side="right", padx=(8, 0))
        self.reset_btn.bind("<Button-1>", lambda e: self._reset_defaults())

        self.import_btn = tk.Label(
            btn_box, text="↑ Import JSON", font=("Segoe UI", 8, "bold"),
            padx=10, pady=7, cursor="hand2"
        )
        self.import_btn.pack(side="right", padx=(8, 0))
        self.import_btn.bind("<Button-1>", lambda e: self._import_json())

        self.export_btn = tk.Label(
            btn_box, text="↓ Export JSON", font=("Segoe UI", 8, "bold"),
            padx=10, pady=7, cursor="hand2"
        )
        self.export_btn.pack(side="right")
        self.export_btn.bind("<Button-1>", lambda e: self._export_json())

        # Two-column layout: Left = Sliders, Right = ROI & Persistence Summary
        self.body_grid = tk.Frame(self.wrapper)
        self.body_grid.pack(fill="both", expand=True)

        self.left_col = tk.Frame(self.body_grid)
        self.left_col.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.right_col = tk.Frame(self.body_grid, width=390)
        self.right_col.pack(side="right", fill="y")
        self.right_col.pack_propagate(False)

        # Slider 1: Lighting Sensitivity (0.0 - 1.0 -> displayed as 0% - 100%)
        self.lighting_slider, self.lighting_val_lbl = self._build_slider_card(
            self.left_col,
            title="Lighting Sensitivity (Brightness & Gamma Boost)",
            desc="Compensates for dim lecture halls (< 0.50) or bright stage lights (> 0.50). 0.50 = Neutral.",
            key="lighting_sensitivity",
            min_val=0.0, max_val=1.0, default=0.50, step=0.05,
            fmt_fn=lambda v: f"Current: {int(round(v * 100))}% ({v:.2f})"
        )

        # Slider 2: Minimum Detection Confidence (0.40 - 0.95)
        self.conf_slider, self.conf_val_lbl = self._build_slider_card(
            self.left_col,
            title="Minimum Detection Confidence Threshold",
            desc="Higher values reduce false positives; lower values improve detection at longer distances.",
            key="min_detection_confidence",
            min_val=0.40, max_val=0.95, default=0.65, step=0.05,
            fmt_fn=lambda v: f"Current: {v:.2f} ({int(round(v * 100))}%)"
        )

        # Toggle Card: Auto-Lighting + CLAHE
        self.auto_card = tk.Frame(self.left_col, padx=18, pady=12, highlightthickness=1)
        self.auto_card.pack(fill="x", pady=(0, 10))
        self.cards.append(self.auto_card)

        self.auto_light_var = tk.BooleanVar(value=bool(self.config.get("enable_auto_lighting_optional", False)))
        self.clahe_var = tk.BooleanVar(value=bool(self.config.get("low_light_clahe", False)))
        self.tremor_var = tk.BooleanVar(value=bool(self.config.get("accessibility_tremor_filter", False)))

        row_a = tk.Frame(self.auto_card)
        row_a.pack(fill="x", pady=2)
        self.auto_chk = tk.Checkbutton(
            row_a, text="Enable Auto-Lighting Adaptive Exposure",
            variable=self.auto_light_var, font=("Segoe UI", 9, "bold"),
            command=self._on_live_param_change, cursor="hand2"
        )
        self.auto_chk.pack(side="left")

        row_b = tk.Frame(self.auto_card)
        row_b.pack(fill="x", pady=2)
        self.clahe_chk = tk.Checkbutton(
            row_b, text="Enable CLAHE Local Contrast Enhancement (Low-Light Mode)",
            variable=self.clahe_var, font=("Segoe UI", 9, "bold"),
            command=self._on_live_param_change, cursor="hand2"
        )
        self.clahe_chk.pack(side="left")

        row_c = tk.Frame(self.auto_card)
        row_c.pack(fill="x", pady=2)
        self.tremor_chk = tk.Checkbutton(
            row_c, text="♿ Accessibility Mode: Tremor Stabilization & Deliberate Hold Filter",
            variable=self.tremor_var, font=("Segoe UI", 9, "bold"),
            command=self._on_live_param_change, cursor="hand2"
        )
        self.tremor_chk.pack(side="left")

        # Right Column Card 1: Region of Interest (ROI)
        self.roi_card = tk.Frame(self.right_col, padx=18, pady=14, highlightthickness=1)
        self.roi_card.pack(fill="x", pady=(0, 10))
        self.cards.append(self.roi_card)

        self.roi_head_lbl = tk.Label(
            self.roi_card, text="REGION OF INTEREST (ROI) ZONE",
            font=("Segoe UI", 9, "bold")
        )
        self.roi_head_lbl.pack(anchor="w")

        self.roi_desc_lbl = tk.Label(
            self.roi_card,
            text="Isolates tracking to your presenter zone. Drag corners directly on the Live Presentation video feed.",
            font=("Segoe UI", 8), wraplength=340, justify="left"
        )
        self.roi_desc_lbl.pack(anchor="w", pady=(2, 8))

        self.roi_var = tk.BooleanVar(value=bool(self.config.get("enable_roi_optional", False)))
        self.roi_chk = tk.Checkbutton(
            self.roi_card, text="Activate Presenter ROI Focus Box",
            variable=self.roi_var, font=("Segoe UI", 9, "bold"),
            command=self._on_live_param_change, cursor="hand2"
        )
        self.roi_chk.pack(anchor="w")

        self.roi_coords_lbl = tk.Label(
            self.roi_card, text="", font=("Consolas", 9, "bold"), pady=6
        )
        self.roi_coords_lbl.pack(anchor="w")

        self.roi_reset_small = tk.Label(
            self.roi_card, text="↺ Reset ROI Box to Default", font=("Segoe UI", 8, "bold"),
            padx=10, pady=4, cursor="hand2"
        )
        self.roi_reset_small.pack(anchor="w", pady=(4, 0))
        self.roi_reset_small.bind("<Button-1>", lambda e: self._handle_reset_roi())

        # Right Column Card 2: Settings Persistence Summary (Figure 4.6)
        self.persist_card = tk.Frame(self.right_col, padx=18, pady=14, highlightthickness=1)
        self.persist_card.pack(fill="both", expand=True)
        self.cards.append(self.persist_card)

        self.persist_head_lbl = tk.Label(
            self.persist_card, text="SETTINGS PERSISTENCE STATUS",
            font=("Segoe UI", 9, "bold")
        )
        self.persist_head_lbl.pack(anchor="w", pady=(0, 6))

        self.persist_summary_lbl = tk.Label(
            self.persist_card, text="", font=("Consolas", 8),
            justify="left", anchor="nw"
        )
        self.persist_summary_lbl.pack(fill="both", expand=True)

        self.refresh_from_config()

    def _build_slider_card(self, parent, title, desc, key, min_val, max_val, default, step, fmt_fn):
        card = tk.Frame(parent, padx=18, pady=12, highlightthickness=1)
        card.pack(fill="x", pady=(0, 10))
        self.cards.append(card)

        top = tk.Frame(card)
        top.pack(fill="x")

        t_lbl = tk.Label(top, text=title, font=("Segoe UI", 10, "bold"))
        t_lbl.pack(side="left")

        val = float(self.config.get(key, default))
        v_lbl = tk.Label(top, text=fmt_fn(val), font=("Segoe UI", 9, "bold"))
        v_lbl.pack(side="right")

        d_lbl = tk.Label(card, text=desc, font=("Segoe UI", 8), justify="left")
        d_lbl.pack(anchor="w", pady=(2, 6))

        scale = tk.Scale(
            card, from_=min_val, to=max_val, resolution=step,
            orient="horizontal", length=560, highlightthickness=0,
            showvalue=False, sliderlength=22,
            command=lambda v, lbl=v_lbl, fn=fmt_fn: self._on_slider_moved(v, lbl, fn)
        )
        scale.set(val)
        scale.pack(fill="x", pady=2)
        return scale, v_lbl

    def _on_slider_moved(self, raw_val, label_widget, fmt_fn):
        label_widget.configure(text=fmt_fn(float(raw_val)))
        self._on_live_param_change()

    def _on_live_param_change(self):
        self.config["lighting_sensitivity"] = round(float(self.lighting_slider.get()), 2)
        self.config["min_detection_confidence"] = round(float(self.conf_slider.get()), 2)
        self.config["enable_roi_optional"] = bool(self.roi_var.get())
        self.config["enable_auto_lighting_optional"] = bool(self.auto_light_var.get())
        self.config["low_light_clahe"] = bool(self.clahe_var.get())
        self.config["accessibility_tremor_filter"] = bool(self.tremor_var.get())
        self.refresh_persistence_summary()
        if callable(self.on_change_callback):
            self.on_change_callback()

    def _handle_reset_roi(self):
        if callable(self.on_reset_roi):
            self.on_reset_roi()
        self.refresh_from_config()

    def refresh_persistence_summary(self):
        b = self.config.get("roi_bounds", [0.16, 0.10, 0.84, 0.90])
        x_px, y_px = int(b[0] * 640), int(b[1] * 480)
        w_px, h_px = int((b[2] - b[0]) * 640), int((b[3] - b[1]) * 480)
        self.roi_coords_lbl.configure(text=f"ROI Box: X={x_px} Y={y_px} W={w_px} H={h_px}")

        mappings_count = len(self.config.get("gesture_mappings", {}))
        theme_name = self.config.get("theme", "light").capitalize()
        summary = (
            f"Storage File  : config.json (Auto-Synced)\n"
            f"Sensitivity   : {int(round(self.config.get('lighting_sensitivity', 0.5)*100))}%\n"
            f"Confidence    : {self.config.get('min_detection_confidence', 0.65):.2f}\n"
            f"Auto-Lighting : {'ON' if self.config.get('enable_auto_lighting_optional') else 'OFF'}\n"
            f"ROI Filter    : {'ON' if self.config.get('enable_roi_optional') else 'OFF'}\n"
            f"Tremor Filter : {'ON (Accessibility)' if self.config.get('accessibility_tremor_filter') else 'OFF'}\n"
            f"UI Theme      : {theme_name}\n"
            f"Key Mappings  : {mappings_count} active gestures"
        )
        self.persist_summary_lbl.configure(text=summary)

    def refresh_from_config(self):
        self.config = self.ipc_config.state
        self.lighting_slider.set(float(self.config.get("lighting_sensitivity", 0.50)))
        self.conf_slider.set(float(self.config.get("min_detection_confidence", 0.65)))
        self.roi_var.set(bool(self.config.get("enable_roi_optional", False)))
        self.auto_light_var.set(bool(self.config.get("enable_auto_lighting_optional", False)))
        self.clahe_var.set(bool(self.config.get("low_light_clahe", False)))
        self.tremor_var.set(bool(self.config.get("accessibility_tremor_filter", False)))
        self.refresh_persistence_summary()

    def _save_calibration(self, show_popup=True):
        self._on_live_param_change()
        self.ipc_config.save(self.config)
        if callable(self.on_save_callback):
            self.on_save_callback()
        if show_popup:
            messagebox.showinfo(
                "Calibration Saved",
                "Calibration parameters saved to config.json and applied to the live feed."
            )

    def _reset_defaults(self):
        self.config["lighting_sensitivity"] = 0.50
        self.config["min_detection_confidence"] = 0.65
        self.config["enable_roi_optional"] = False
        self.config["enable_auto_lighting_optional"] = False
        self.config["low_light_clahe"] = False
        self.config["accessibility_tremor_filter"] = False
        self.config["roi_bounds"] = [0.16, 0.10, 0.84, 0.90]
        self.ipc_config.save(self.config)
        self.refresh_from_config()
        if callable(self.on_change_callback):
            self.on_change_callback()
        messagebox.showinfo("Calibration Reset", "All calibration values restored to factory defaults.")

    def _export_json(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON Configuration", "*.json")],
            initialfile="wave_calibration_export.json"
        )
        if path:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=4)
            messagebox.showinfo("Export Settings", f"Configuration exported to:\n{path}")

    def _import_json(self):
        path = filedialog.askopenfilename(filetypes=[("JSON Configuration", "*.json")])
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                imported = json.load(f)
            if not isinstance(imported, dict):
                raise ValueError("Invalid JSON root object")
            self.config.update(imported)
            self.ipc_config.save(self.config)
            self.refresh_from_config()
            if callable(self.on_change_callback):
                self.on_change_callback()
            messagebox.showinfo("Import Settings", "Configuration imported and applied successfully.")
        except Exception as e:
            messagebox.showerror("Import Error", f"Could not import JSON file:\n{e}")

    def apply_theme(self, palette):
        self.theme = palette
        bg_main = palette["BG_MAIN"]
        bg_card = palette["BG_CARD"]
        border = palette["BORDER"]
        fg_primary = palette["FG_PRIMARY"]
        fg_muted = palette["FG_MUTED"]
        accent = palette["ACCENT"]
        pill_bg = palette["PILL_BG"]

        self.configure(bg=bg_main)
        for frame in (self.wrapper, self.header, self.body_grid, self.left_col, self.right_col):
            frame.configure(bg=bg_main)
        for sub in self.header.winfo_children():
            sub.configure(bg=bg_main)

        self.kicker_lbl.configure(bg=bg_main, fg=accent)
        self.title_lbl.configure(bg=bg_main, fg=fg_primary)
        self.sub_lbl.configure(bg=bg_main, fg=fg_muted)

        self.save_btn.configure(bg=accent, fg="#FFFFFF")
        self.reset_btn.configure(bg=bg_card, fg=fg_primary, highlightthickness=1, highlightbackground=border)
        self.import_btn.configure(bg=bg_card, fg=fg_primary, highlightthickness=1, highlightbackground=border)
        self.export_btn.configure(bg=bg_card, fg=fg_primary, highlightthickness=1, highlightbackground=border)

        for card in self.cards:
            card.configure(bg=bg_card, highlightbackground=border)
            for child in card.winfo_children():
                if isinstance(child, tk.Frame):
                    child.configure(bg=bg_card)
                    for grand in child.winfo_children():
                        if isinstance(grand, tk.Label):
                            grand.configure(bg=bg_card, fg=accent if "Current:" in grand.cget("text") else fg_primary)
                        elif isinstance(grand, tk.Checkbutton):
                            grand.configure(bg=bg_card, fg=fg_primary, activebackground=bg_card,
                                            activeforeground=fg_primary, selectcolor=bg_card)
                elif isinstance(child, tk.Label):
                    child.configure(bg=bg_card, fg=fg_muted if len(child.cget("text")) > 45 else fg_primary)
                elif isinstance(child, tk.Checkbutton):
                    child.configure(bg=bg_card, fg=fg_primary, activebackground=bg_card,
                                    activeforeground=fg_primary, selectcolor=bg_card)
                elif isinstance(child, tk.Scale):
                    child.configure(bg=bg_card, troughcolor=pill_bg, fg=accent)

        self.roi_head_lbl.configure(fg=accent, bg=bg_card)
        self.persist_head_lbl.configure(fg=accent, bg=bg_card)
        self.roi_coords_lbl.configure(fg=accent, bg=bg_card)
        self.persist_summary_lbl.configure(fg=fg_primary, bg=bg_card)
        self.roi_reset_small.configure(bg=pill_bg, fg=fg_primary)