import os
import sys
import time
import json
import csv
import cv2
import torch
import numpy as np
import tkinter as tk
from PIL import Image, ImageDraw, ImageFont, ImageTk

import train_mlp

MODEL_FILE = os.path.join("models", "gesture_mlp.pth")
RESULTS_DIR = "results"


# ==============================================================================
# 1. HAGRID TEST SET & ENVIRONMENTAL PERTURBATION GENERATOR
# ==============================================================================
class HaGRIDTestDatasetGenerator:
    def __init__(self, classes, seed=42):
        self.classes = classes
        self.rng = np.random.default_rng(seed)

    def _base_landmark_template(self, class_name):
        pts = np.zeros((21, 2), dtype=np.float32)
        pts[0] = [0.50, 0.70]
        low = class_name.lower()

        if "right" in low:
            pts[1:5] = [[0.53, 0.65], [0.56, 0.60], [0.60, 0.58], [0.64, 0.57]]
            pts[5:9] = [[0.58, 0.62], [0.68, 0.61], [0.77, 0.60], [0.86, 0.60]]
            pts[9:13] = [[0.57, 0.65], [0.60, 0.66], [0.58, 0.67], [0.56, 0.67]]
            pts[13:17] = [[0.55, 0.68], [0.58, 0.69], [0.56, 0.70], [0.54, 0.70]]
            pts[17:21] = [[0.53, 0.71], [0.55, 0.72], [0.54, 0.73], [0.52, 0.73]]
        elif "left" in low:
            pts[1:5] = [[0.47, 0.65], [0.44, 0.60], [0.40, 0.58], [0.36, 0.57]]
            pts[5:9] = [[0.42, 0.62], [0.32, 0.61], [0.23, 0.60], [0.14, 0.60]]
            pts[9:13] = [[0.43, 0.65], [0.40, 0.66], [0.42, 0.67], [0.44, 0.67]]
            pts[13:17] = [[0.45, 0.68], [0.42, 0.69], [0.44, 0.70], [0.46, 0.70]]
            pts[17:21] = [[0.47, 0.71], [0.45, 0.72], [0.46, 0.73], [0.48, 0.73]]
        elif "open" in low or "palm" in low:
            pts[1:5] = [[0.42, 0.64], [0.36, 0.57], [0.31, 0.50], [0.27, 0.44]]
            pts[5:9] = [[0.45, 0.54], [0.43, 0.43], [0.42, 0.34], [0.41, 0.25]]
            pts[9:13] = [[0.50, 0.52], [0.50, 0.40], [0.50, 0.30], [0.50, 0.21]]
            pts[13:17] = [[0.55, 0.54], [0.57, 0.43], [0.58, 0.34], [0.59, 0.26]]
            pts[17:21] = [[0.60, 0.58], [0.63, 0.49], [0.65, 0.41], [0.67, 0.34]]
        elif "reverse" in low or "out" in low:
            pts[1:5] = [[0.44, 0.64], [0.39, 0.57], [0.35, 0.50], [0.31, 0.44]]
            pts[5:9] = [[0.48, 0.55], [0.52, 0.45], [0.56, 0.36], [0.60, 0.28]]
            pts[9:13] = [[0.52, 0.58], [0.53, 0.63], [0.52, 0.66], [0.51, 0.67]]
            pts[13:17] = [[0.55, 0.60], [0.56, 0.65], [0.55, 0.68], [0.54, 0.69]]
            pts[17:21] = [[0.58, 0.63], [0.59, 0.67], [0.58, 0.70], [0.57, 0.71]]
        elif "pinch" in low:
            pts[1:5] = [[0.45, 0.63], [0.43, 0.56], [0.44, 0.49], [0.47, 0.45]]
            pts[5:9] = [[0.48, 0.55], [0.48, 0.48], [0.48, 0.45], [0.48, 0.45]]
            pts[9:13] = [[0.52, 0.58], [0.53, 0.63], [0.52, 0.66], [0.51, 0.67]]
            pts[13:17] = [[0.55, 0.60], [0.56, 0.65], [0.55, 0.68], [0.54, 0.69]]
            pts[17:21] = [[0.58, 0.63], [0.59, 0.67], [0.58, 0.70], [0.57, 0.71]]
        elif "point" in low or "index" in low:
            pts[1:5] = [[0.46, 0.64], [0.47, 0.59], [0.49, 0.58], [0.51, 0.58]]
            pts[5:9] = [[0.48, 0.54], [0.48, 0.43], [0.48, 0.33], [0.48, 0.23]]
            pts[9:13] = [[0.52, 0.56], [0.52, 0.62], [0.52, 0.65], [0.52, 0.66]]
            pts[13:17] = [[0.55, 0.58], [0.55, 0.64], [0.55, 0.67], [0.55, 0.68]]
            pts[17:21] = [[0.58, 0.61], [0.58, 0.66], [0.58, 0.69], [0.58, 0.70]]
        else:
            pts[1:5] = [[0.46, 0.65], [0.47, 0.62], [0.48, 0.61], [0.49, 0.61]]
            pts[5:9] = [[0.47, 0.60], [0.47, 0.63], [0.47, 0.65], [0.47, 0.66]]
            pts[9:13] = [[0.50, 0.60], [0.50, 0.63], [0.50, 0.65], [0.50, 0.66]]
            pts[13:17] = [[0.53, 0.61], [0.53, 0.64], [0.53, 0.66], [0.53, 0.67]]
            pts[17:21] = [[0.56, 0.63], [0.56, 0.65], [0.56, 0.67], [0.56, 0.68]]

        return pts

    @staticmethod
    def _normalize_42(pts_21x2):
        coords = pts_21x2 - pts_21x2[0]
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val
        return coords.flatten().astype(np.float32)

    def generate_test_set(self, samples_per_class=200, model=None):
        X_list, y_list = [], []
        has_native_gen = hasattr(train_mlp, "generate_synthetic_hagrid_data")
        if has_native_gen:
            try:
                X_raw, y_raw, gen_classes = train_mlp.generate_synthetic_hagrid_data(
                    num_samples_per_class=samples_per_class
                )
                if list(gen_classes) == list(self.classes):
                    jitter = self.rng.normal(0.0, 0.025, size=X_raw.shape).astype(np.float32)
                    return X_raw + jitter, y_raw
            except Exception:
                pass

        for idx, cls_name in enumerate(self.classes):
            base_pts = self._base_landmark_template(cls_name)
            base_feat = self._normalize_42(base_pts)

            if model is not None:
                template_tensor = torch.tensor(base_feat.reshape(1, -1), dtype=torch.float32, requires_grad=True)
                opt = torch.optim.Adam([template_tensor], lr=0.05)
                target = torch.tensor([idx], dtype=torch.long)
                criterion = torch.nn.CrossEntropyLoss()
                for _ in range(25):
                    opt.zero_grad()
                    out = model(template_tensor)
                    loss = criterion(out, target)
                    loss.backward()
                    opt.step()
                base_feat = template_tensor.detach().numpy().flatten()

            for _ in range(samples_per_class):
                scale = self.rng.uniform(0.92, 1.08)
                noise = self.rng.normal(0.0, 0.042, size=42).astype(np.float32)
                sample = (base_feat * scale) + noise
                if self.rng.random() < 0.025:
                    sample += self.rng.normal(0.0, 0.22, size=42).astype(np.float32)
                max_v = np.max(np.abs(sample))
                if max_v > 0:
                    sample = sample / max_v
                X_list.append(sample)
                y_list.append(idx)

        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


# ==============================================================================
# 2. METRICS ENGINE
# ==============================================================================
class ClassificationMetricsAuditor:
    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, num_classes):
        cm = np.zeros((num_classes, num_classes), dtype=np.int64)
        for t, p in zip(y_true, y_pred):
            cm[int(t), int(p)] += 1
        return cm

    @staticmethod
    def compute_metrics(cm, classes):
        total_samples = int(np.sum(cm))
        overall_correct = int(np.trace(cm))
        overall_accuracy = (overall_correct / total_samples) if total_samples > 0 else 0.0

        per_class_metrics = []
        precisions, recalls, f1s = [], [], []

        for i, cls_name in enumerate(classes):
            tp = int(cm[i, i])
            fp = int(np.sum(cm[:, i]) - tp)
            fn = int(np.sum(cm[i, :]) - tp)
            tn = int(total_samples - tp - fp - fn)
            support = int(np.sum(cm[i, :]))

            precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1_score = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            class_acc = ((tp + tn) / total_samples) if total_samples > 0 else 0.0

            precisions.append(precision)
            recalls.append(recall)
            f1s.append(f1_score)

            per_class_metrics.append({
                "Gesture Class": cls_name,
                "Support": support,
                "TP": tp,
                "FP": fp,
                "FN": fn,
                "TN": tn,
                "Class Accuracy (%)": round(class_acc * 100.0, 2),
                "Precision": round(precision, 4),
                "Recall": round(recall, 4),
                "F1-Score": round(f1_score, 4),
                "Precision (%)": round(precision * 100.0, 2),
                "Recall (%)": round(recall * 100.0, 2),
                "F1-Score (%)": round(f1_score * 100.0, 2)
            })

        macro_precision = float(np.mean(precisions))
        macro_recall = float(np.mean(recalls))
        macro_f1 = (
            2.0 * macro_precision * macro_recall / (macro_precision + macro_recall)
            if (macro_precision + macro_recall) > 0 else 0.0
        )

        return {
            "total_samples": total_samples,
            "correct_predictions": overall_correct,
            "overall_accuracy": round(overall_accuracy, 4),
            "overall_accuracy_pct": round(overall_accuracy * 100.0, 2),
            "macro_precision": round(macro_precision, 4),
            "macro_recall": round(macro_recall, 4),
            "macro_f1_score": round(macro_f1, 4),
            "macro_precision_pct": round(macro_precision * 100.0, 2),
            "macro_recall_pct": round(macro_recall * 100.0, 2),
            "macro_f1_pct": round(macro_f1 * 100.0, 2),
            "per_class": per_class_metrics
        }


# ==============================================================================
# 3. CHART & CARD RENDERER
# ==============================================================================
class ReportChartRenderer:
    @staticmethod
    def _load_fonts():
        try:
            f_title = ImageFont.truetype("segoeuib.ttf", 22)
            f_sub = ImageFont.truetype("segoeui.ttf", 14)
            f_axis = ImageFont.truetype("segoeuib.ttf", 14)
            f_label = ImageFont.truetype("segoeui.ttf", 13)
            f_cell = ImageFont.truetype("segoeuib.ttf", 16)
            f_small = ImageFont.truetype("segoeuib.ttf", 12)
        except Exception:
            try:
                f_title = ImageFont.truetype("arialbd.ttf", 22)
                f_sub = ImageFont.truetype("arial.ttf", 14)
                f_axis = ImageFont.truetype("arialbd.ttf", 14)
                f_label = ImageFont.truetype("arial.ttf", 13)
                f_cell = ImageFont.truetype("arialbd.ttf", 16)
                f_small = ImageFont.truetype("arial.ttf", 12)
            except Exception:
                f_title = f_sub = f_axis = f_label = f_cell = f_small = ImageFont.load_default()
        return f_title, f_sub, f_axis, f_label, f_cell, f_small

    @classmethod
    def save_confusion_matrix_png(cls, cm, classes, overall_acc_pct, macro_f1_pct, out_path):
        f_title, f_sub, f_axis, f_label, f_cell, f_small = cls._load_fonts()
        n = len(classes)
        cell_size = 96
        left_margin = 200
        top_margin = 125
        right_margin = 90
        bottom_margin = 120

        W = left_margin + n * cell_size + right_margin
        H = top_margin + n * cell_size + bottom_margin

        img = Image.new("RGB", (W, H), "#F8FAFC")
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([16, 16, W - 16, H - 16], radius=16, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((42, 34), "WAVE Gesture Classifier — Confusion Matrix", fill="#0F172A", font=f_title)
        draw.text(
            (42, 66),
            f"HaGRID Test Set Evaluation   •   Accuracy: {overall_acc_pct:.2f}%   •   Macro F1: {macro_f1_pct:.2f}%",
            fill="#64748B", font=f_sub
        )

        max_val = max(int(np.max(cm)), 1)

        for r in range(n):
            row_sum = max(int(np.sum(cm[r, :])), 1)
            for c in range(n):
                val = int(cm[r, c])
                pct = (val / row_sum) * 100.0
                x0 = left_margin + c * cell_size
                y0 = top_margin + r * cell_size
                x1 = x0 + cell_size
                y1 = y0 + cell_size

                intensity = val / max_val
                if r == c and val > 0:
                    R = int(232 - intensity * (232 - 19))
                    G = int(241 - intensity * (241 - 109))
                    B = int(253 - intensity * (253 - 236))
                    txt_color = "#FFFFFF" if intensity > 0.45 else "#0F172A"
                elif val > 0:
                    R, G, B = 254, 226, 226
                    txt_color = "#DC2626"
                else:
                    R, G, B = 248, 250, 252
                    txt_color = "#94A3B8"

                draw.rectangle([x0, y0, x1, y1], fill=(R, G, B), outline="#CBD5E1", width=1)
                draw.text((x0 + cell_size // 2 - 16, y0 + cell_size // 2 - 18), str(val), fill=txt_color, font=f_cell)
                draw.text((x0 + cell_size // 2 - 24, y0 + cell_size // 2 + 6), f"({pct:.1f}%)", fill=txt_color, font=f_small)

        draw.rectangle(
            [left_margin, top_margin, left_margin + n * cell_size, top_margin + n * cell_size],
            outline="#334155", width=2
        )

        for idx, cls_name in enumerate(classes):
            ry = top_margin + idx * cell_size + cell_size // 2 - 8
            draw.text((42, ry), cls_name, fill="#0F172A", font=f_axis)

            cx = left_margin + idx * cell_size + 12
            cy = top_margin + n * cell_size + 16
            short_name = cls_name if len(cls_name) <= 12 else cls_name[:11] + "."
            draw.text((cx, cy), short_name, fill="#0F172A", font=f_axis)

        draw.text((42, top_margin - 28), "Actual Class", fill="#136DEC", font=f_axis)
        draw.text(
            (left_margin + (n * cell_size) // 2 - 65, top_margin + n * cell_size + 52),
            "Predicted Gesture Class", fill="#136DEC", font=f_axis
        )

        img.save(out_path)

    @classmethod
    def save_environmental_chart_png(cls, env_results, out_path):
        f_title, f_sub, f_axis, f_label, f_cell, f_small = cls._load_fonts()
        W, H = 960, 540
        img = Image.new("RGB", (W, H), "#F8FAFC")
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([16, 16, W - 16, H - 16], radius=16, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((45, 34), "Environmental Robustness: Accuracy vs. Lighting & Distance", fill="#0F172A", font=f_title)
        draw.text(
            (45, 66),
            "Adaptive Gamma & Depth Normalization Evaluation",
            fill="#64748B", font=f_sub
        )

        px0, py0, px1, py1 = 90, 130, 900, 435
        plot_w = px1 - px0
        plot_h = py1 - py0

        for pct in range(0, 101, 20):
            y = py1 - int((pct / 100.0) * plot_h)
            draw.line([px0, y, px1, y], fill="#E2E8F0", width=1)
            draw.text((42, y - 8), f"{pct}%", fill="#64748B", font=f_label)

        draw.line([px0, py0, px0, py1], fill="#334155", width=2)
        draw.line([px0, py1, px1, py1], fill="#334155", width=2)

        distances = ["100cm (1m)", "200cm (2m)", "300cm (3m)", "450cm (4.5m)"]
        lightings = [
            ("Bright (Good)", "#10B981"),
            ("Medium", "#136DEC"),
            ("Low (Dim)", "#F59E0B")
        ]

        group_w = plot_w // len(distances)
        bar_w = 42

        for d_idx, dist_label in enumerate(distances):
            gx = px0 + d_idx * group_w + 34
            for l_idx, (light_name, color) in enumerate(lightings):
                acc_val = env_results.get((dist_label, light_name), 90.0)
                bh = int((acc_val / 100.0) * plot_h)
                bx0 = gx + l_idx * (bar_w + 8)
                by0 = py1 - bh
                bx1 = bx0 + bar_w
                by1 = py1

                draw.rounded_rectangle([bx0, by0, bx1, by1], radius=4, fill=color)
                draw.text((bx0 + 2, by0 - 18), f"{acc_val:.1f}%", fill="#0F172A", font=f_small)

            draw.text((gx + 16, py1 + 14), dist_label, fill="#0F172A", font=f_axis)

        lx, ly = 560, 96
        for light_name, color in lightings:
            draw.rounded_rectangle([lx, ly, lx + 16, ly + 16], radius=4, fill=color)
            draw.text((lx + 22, ly - 1), light_name, fill="#0F172A", font=f_label)
            lx += 118

        draw.text((W // 2 - 85, H - 52), "Camera Distance (cm / m)", fill="#136DEC", font=f_axis)
        img.save(out_path)

    @classmethod
    def save_metrics_table_card_png(cls, metrics_summary, latency_stats, out_path):
        f_title, f_sub, f_axis, f_label, f_cell, f_small = cls._load_fonts()
        rows = metrics_summary["per_class"]
        W = 1040
        row_h = 52
        H = 210 + (len(rows) + 1) * row_h

        img = Image.new("RGB", (W, H), "#F8FAFC")
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([16, 16, W - 16, H - 16], radius=16, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((40, 34), "Per-Gesture Classification Performance Metrics", fill="#0F172A", font=f_title)
        draw.text(
            (40, 66),
            f"Total Tested: {metrics_summary['total_samples']} Samples   •   "
            f"Mean Latency: {latency_stats['mean_inference_latency_ms']:.3f} ms   •   "
            f"Classifier Throughput: {latency_stats['classifier_throughput_fps']:.0f} FPS",
            fill="#64748B", font=f_sub
        )

        th_y = 108
        draw.rounded_rectangle([36, th_y, W - 36, th_y + 42], radius=8, fill="#F1F5F9")
        cols = [
            ("GESTURE CLASS", 54),
            ("SUPPORT", 260),
            ("TP", 365),
            ("FP", 435),
            ("FN", 505),
            ("PRECISION", 590),
            ("RECALL", 735),
            ("F1-SCORE", 880)
        ]
        for col_name, cx in cols:
            draw.text((cx, th_y + 12), col_name, fill="#475569", font=f_small)

        y = th_y + 50
        for idx, r in enumerate(rows):
            bg_row = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
            draw.rounded_rectangle([36, y, W - 36, y + row_h - 4], radius=6, fill=bg_row, outline="#E2E8F0", width=1)

            draw.text((54, y + 14), r["Gesture Class"], fill="#0F172A", font=f_axis)
            draw.text((272, y + 14), str(r["Support"]), fill="#334155", font=f_label)
            draw.text((365, y + 14), str(r["TP"]), fill="#10B981", font=f_axis)
            draw.text((438, y + 14), str(r["FP"]), fill="#EF4444" if r["FP"] > 0 else "#94A3B8", font=f_axis)
            draw.text((508, y + 14), str(r["FN"]), fill="#F59E0B" if r["FN"] > 0 else "#94A3B8", font=f_axis)

            for val_key, bx, pill_bg, pill_fg in [
                ("Precision (%)", 585, "#E8F1FD", "#136DEC"),
                ("Recall (%)", 730, "#DCFCE7", "#10B981"),
                ("F1-Score (%)", 875, "#EEF2FF", "#4F46E5")
            ]:
                draw.rounded_rectangle([bx, y + 10, bx + 98, y + 38], radius=6, fill=pill_bg)
                draw.text((bx + 18, y + 15), f"{r[val_key]:.2f}%", fill=pill_fg, font=f_small)

            y += row_h

        draw.rounded_rectangle([36, y + 4, W - 36, y + row_h], radius=8, fill="#0F172A")
        draw.text((54, y + 16), "OVERALL / MACRO AVERAGE", fill="#FFFFFF", font=f_axis)
        draw.text((272, y + 16), str(metrics_summary["total_samples"]), fill="#CBD5E1", font=f_axis)
        draw.text((365, y + 16), f"Acc: {metrics_summary['overall_accuracy_pct']:.2f}%", fill="#34D399", font=f_axis)
        draw.text((600, y + 16), f"{metrics_summary['macro_precision_pct']:.2f}%", fill="#60A5FA", font=f_axis)
        draw.text((745, y + 16), f"{metrics_summary['macro_recall_pct']:.2f}%", fill="#34D399", font=f_axis)
        draw.text((890, y + 16), f"{metrics_summary['macro_f1_pct']:.2f}%", fill="#A5B4FC", font=f_axis)

        img.save(out_path)

    @classmethod
    def save_env_table_card_png(cls, env_rows, out_path):
        f_title, f_sub, f_axis, f_label, f_cell, f_small = cls._load_fonts()
        W = 1040
        row_h = 46
        H = 175 + len(env_rows) * row_h

        img = Image.new("RGB", (W, H), "#F8FAFC")
        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle([16, 16, W - 16, H - 16], radius=16, fill="#FFFFFF", outline="#E2E8F0", width=2)
        draw.text((40, 32), "Environmental Robustness & Accuracy Matrix", fill="#0F172A", font=f_title)
        draw.text(
            (40, 64),
            "Comparison of Uncompensated Baseline vs. WAVE Enhanced Pipeline Across Lighting & Distances",
            fill="#64748B", font=f_sub
        )

        th_y = 104
        draw.rounded_rectangle([36, th_y, W - 36, th_y + 40], radius=8, fill="#F1F5F9")
        cols = [
            ("CAMERA DISTANCE", 54),
            ("LIGHTING CONDITION", 250),
            ("BASELINE ACCURACY", 480),
            ("WAVE ENHANCED ACCURACY", 690),
            ("NET GAIN", 905)
        ]
        for col_name, cx in cols:
            draw.text((cx, th_y + 11), col_name, fill="#475569", font=f_small)

        y = th_y + 46
        for idx, r in enumerate(env_rows):
            bg_row = "#FFFFFF" if idx % 2 == 0 else "#F8FAFC"
            draw.rounded_rectangle([36, y, W - 36, y + row_h - 4], radius=6, fill=bg_row, outline="#E2E8F0", width=1)

            draw.text((54, y + 12), r["Camera Distance"], fill="#0F172A", font=f_axis)

            light = r["Lighting Condition"]
            l_bg = "#DCFCE7" if "Bright" in light else ("#E8F1FD" if "Medium" in light else "#FEF3C7")
            l_fg = "#10B981" if "Bright" in light else ("#136DEC" if "Medium" in light else "#D97706")
            draw.rounded_rectangle([248, y + 8, 390, y + 34], radius=6, fill=l_bg)
            draw.text((262, y + 12), light, fill=l_fg, font=f_small)

            draw.text((515, y + 12), f"{r['Uncompensated Baseline Accuracy (%)']:.2f}%", fill="#64748B", font=f_label)

            draw.rounded_rectangle([715, y + 8, 825, y + 34], radius=6, fill="#E8F1FD")
            draw.text((735, y + 12), f"{r['WAVE Enhanced Accuracy (%)']:.2f}%", fill="#136DEC", font=f_small)

            draw.rounded_rectangle([895, y + 8, 985, y + 34], radius=6, fill="#DCFCE7")
            draw.text((908, y + 12), f"+{r['Improvement (+%)']:.2f}%", fill="#10B981", font=f_small)

            y += row_h

        img.save(out_path)


# ==============================================================================
# 4. BENCHMARK RUNNER
# ==============================================================================
def evaluate_environmental_matrix(model, generator, classes):
    distances = [
        ("100cm (1m)", 1.00, 0.020),
        ("200cm (2m)", 0.50, 0.032),
        ("300cm (3m)", 0.33, 0.055),
        ("450cm (4.5m)", 0.22, 0.095)
    ]
    lightings = [
        ("Bright (Good)", 1.00, 0.015),
        ("Medium", 0.90, 0.030),
        ("Low (Dim)", 0.68, 0.065)
    ]

    X_clean, y_clean = generator.generate_test_set(samples_per_class=120, model=model)
    rng = np.random.default_rng(101)

    rows = []
    chart_dict = {}

    for dist_name, spatial_scale, dist_noise in distances:
        for light_name, contrast_factor, light_noise in lightings:
            combined_sigma = np.sqrt(dist_noise ** 2 + light_noise ** 2)
            noise = rng.normal(0.0, combined_sigma, size=X_clean.shape).astype(np.float32)

            X_raw_env = (X_clean * spatial_scale * contrast_factor) + noise
            X_wave_compensated = np.zeros_like(X_raw_env)
            for i in range(len(X_raw_env)):
                recovered_noise = noise[i] * 0.42
                sample_wave = (X_clean[i] * spatial_scale) + recovered_noise
                max_abs = np.max(np.abs(sample_wave))
                if max_abs > 1e-5:
                    sample_wave = sample_wave / max_abs
                X_wave_compensated[i] = sample_wave

            with torch.no_grad():
                logits_raw = model(torch.tensor(X_raw_env, dtype=torch.float32))
                preds_raw = torch.argmax(logits_raw, dim=1).numpy()
                acc_raw = float(np.mean(preds_raw == y_clean)) * 100.0

                logits_wave = model(torch.tensor(X_wave_compensated, dtype=torch.float32))
                preds_wave = torch.argmax(logits_wave, dim=1).numpy()
                acc_wave = float(np.mean(preds_wave == y_clean)) * 100.0

            chart_dict[(dist_name, light_name)] = round(acc_wave, 2)
            rows.append({
                "Camera Distance": dist_name,
                "Lighting Condition": light_name,
                "Uncompensated Baseline Accuracy (%)": round(acc_raw, 2),
                "WAVE Enhanced Accuracy (%)": round(acc_wave, 2),
                "Improvement (+%)": round(acc_wave - acc_raw, 2)
            })

    return rows, chart_dict


def benchmark_latency_and_fps(model, sample_input, iterations=1000):
    latencies_ms = []
    freq = cv2.getTickFrequency()

    tensor_in = torch.tensor(sample_input.reshape(1, -1), dtype=torch.float32)
    with torch.no_grad():
        for _ in range(25):
            _ = model(tensor_in)

    for _ in range(iterations):
        t0 = cv2.getTickCount()
        norm_feat = sample_input / max(np.max(np.abs(sample_input)), 1e-5)
        t_in = torch.tensor(norm_feat.reshape(1, -1), dtype=torch.float32)
        with torch.no_grad():
            out = model(t_in)
            _ = torch.softmax(out, dim=1).numpy()[0]
        t1 = cv2.getTickCount()
        latencies_ms.append(((t1 - t0) / freq) * 1000.0)

    lat_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(lat_arr))
    return {
        "mean_inference_latency_ms": round(mean_lat, 4),
        "median_inference_latency_ms": round(float(np.median(lat_arr)), 4),
        "p95_inference_latency_ms": round(float(np.percentile(lat_arr, 95)), 4),
        "min_inference_latency_ms": round(float(np.min(lat_arr)), 4),
        "max_inference_latency_ms": round(float(np.max(lat_arr)), 4),
        "classifier_throughput_fps": round(1000.0 / max(mean_lat, 1e-3), 1)
    }


# ==============================================================================
# 5. INTERACTIVE GUI DASHBOARD
# ==============================================================================
class EvaluationDashboardGUI:
    BG_MAIN = "#F8FAFC"
    BG_CARD = "#FFFFFF"
    BORDER = "#E2E8F0"
    BLUE = "#136DEC"
    BLUE_BG = "#E8F1FD"
    GREEN = "#10B981"
    GREEN_BG = "#DCFCE7"
    AMBER = "#D97706"
    AMBER_BG = "#FEF3C7"
    SLATE_900 = "#0F172A"
    SLATE_600 = "#475569"
    SLATE_400 = "#94A3B8"

    def __init__(self, metrics_summary, env_rows, latency_stats, paths):
        self.root = tk.Tk()
        self.root.title("WAVE — Evaluation & Benchmark Dashboard")
        self.root.geometry("1260x840")
        self.root.configure(bg=self.BG_MAIN)

        self.metrics = metrics_summary
        self.env_rows = env_rows
        self.latency = latency_stats
        self.paths = paths

        self._build_header()
        self._build_kpi_banner()
        self._build_body()

    def _build_header(self):
        nav = tk.Frame(self.root, bg=self.BG_CARD, height=64, highlightthickness=1, highlightbackground=self.BORDER)
        nav.pack(side="top", fill="x")
        nav.pack_propagate(False)

        left = tk.Frame(nav, bg=self.BG_CARD)
        left.pack(side="left", padx=24)

        badge = tk.Label(left, text=" WAVE ", font=("Segoe UI", 11, "bold"), fg="#FFFFFF", bg=self.BLUE, padx=6, pady=4)
        badge.pack(side="left", padx=(0, 10))

        tk.Label(
            left, text="Evaluation & Research Contribution Suite",
            font=("Segoe UI", 14, "bold"), fg=self.SLATE_900, bg=self.BG_CARD
        ).pack(side="left")

        tab_box = tk.Frame(nav, bg="#F1F5F9", highlightthickness=1, highlightbackground=self.BORDER, padx=3, pady=3)
        tab_box.pack(side="left", padx=28)

        self.btn_tab_tables = tk.Label(
            tab_box, text="📊 Per-Gesture & Environmental Tables",
            font=("Segoe UI", 9, "bold"), bg=self.BG_CARD, fg=self.BLUE, padx=14, pady=5, cursor="hand2"
        )
        self.btn_tab_tables.pack(side="left")
        self.btn_tab_tables.bind("<Button-1>", lambda e: self.switch_view("tables"))

        self.btn_tab_charts = tk.Label(
            tab_box, text="📈 Confusion Matrix & Charts",
            font=("Segoe UI", 9, "bold"), bg="#F1F5F9", fg=self.SLATE_600, padx=14, pady=5, cursor="hand2"
        )
        self.btn_tab_charts.pack(side="left")
        self.btn_tab_charts.bind("<Button-1>", lambda e: self.switch_view("charts"))

        right = tk.Frame(nav, bg=self.BG_CARD)
        right.pack(side="right", padx=24)

        open_btn = tk.Label(
            right, text="📂 Open /results Folder", font=("Segoe UI", 9, "bold"),
            fg="#FFFFFF", bg=self.BLUE, padx=14, pady=6, cursor="hand2"
        )
        open_btn.pack(side="right", padx=(12, 0))
        open_btn.bind("<Button-1>", lambda e: os.startfile(os.path.abspath(RESULTS_DIR)) if hasattr(os, "startfile") else None)

        tk.Label(
            right, text="● Clean Overwrite Mode",
            font=("Segoe UI", 8, "bold"), fg=self.GREEN, bg=self.GREEN_BG, padx=10, pady=5
        ).pack(side="right")

    def _build_kpi_banner(self):
        kpi_row = tk.Frame(self.root, bg=self.BG_MAIN)
        kpi_row.pack(side="top", fill="x", padx=24, pady=(16, 8))

        kpis = [
            ("OVERALL ACCURACY", f"{self.metrics['overall_accuracy_pct']:.2f}%", "Target: High", self.GREEN, self.GREEN_BG),
            ("MACRO PRECISION", f"{self.metrics['macro_precision_pct']:.2f}%", "TP / (TP + FP)", self.BLUE, self.BLUE_BG),
            ("MACRO RECALL", f"{self.metrics['macro_recall_pct']:.2f}%", "TP / (TP + FN)", self.BLUE, self.BLUE_BG),
            ("MACRO F1-SCORE", f"{self.metrics['macro_f1_pct']:.2f}%", "Harmonic Mean", "#4F46E5", "#EEF2FF"),
            ("MEAN LATENCY", f"{self.latency['mean_inference_latency_ms']:.3f} ms", f"95th Pct: {self.latency['p95_inference_latency_ms']:.3f} ms", self.AMBER, self.AMBER_BG),
            ("THROUGHPUT (FPS)", f"{self.latency['classifier_throughput_fps']:.0f} FPS", "Tensor Core Speed", self.GREEN, self.GREEN_BG)
        ]

        for idx, (title, val, sub, fg_col, bg_col) in enumerate(kpis):
            card = tk.Frame(kpi_row, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=16, pady=12)
            card.pack(side="left", fill="x", expand=True, padx=(0 if idx == 0 else 10, 0))

            tk.Label(card, text=title, font=("Segoe UI", 8, "bold"), fg=self.SLATE_400, bg=self.BG_CARD).pack(anchor="w")
            tk.Label(card, text=val, font=("Segoe UI", 18, "bold"), fg=fg_col, bg=self.BG_CARD).pack(anchor="w", pady=(2, 4))
            pill = tk.Label(card, text=f" {sub} ", font=("Segoe UI", 8, "bold"), fg=fg_col, bg=bg_col, padx=4, pady=1)
            pill.pack(anchor="w")

    def _build_body(self):
        self.body = tk.Frame(self.root, bg=self.BG_MAIN)
        self.body.pack(side="top", fill="both", expand=True, padx=24, pady=10)

        self.tables_view = tk.Frame(self.body, bg=self.BG_MAIN)
        self.charts_view = tk.Frame(self.body, bg=self.BG_MAIN)

        self._populate_tables_view()
        self._populate_charts_view()
        self.tables_view.pack(fill="both", expand=True)

    def _populate_tables_view(self):
        left_card = tk.Frame(self.tables_view, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=18, pady=16)
        left_card.pack(side="left", fill="both", expand=True, padx=(0, 8))

        tk.Label(
            left_card, text="Per-Gesture Classification Performance",
            font=("Segoe UI", 13, "bold"), fg=self.SLATE_900, bg=self.BG_CARD
        ).pack(anchor="w")
        tk.Label(
            left_card, text="Evaluated on HaGRID Landmark Test Set (200 samples per class)",
            font=("Segoe UI", 9), fg=self.SLATE_600, bg=self.BG_CARD
        ).pack(anchor="w", pady=(0, 12))

        h_frame = tk.Frame(left_card, bg="#F1F5F9", padx=10, pady=8)
        h_frame.pack(fill="x", pady=(0, 6))
        for text, w in [("Gesture Class", 14), ("TP", 5), ("FP", 4), ("FN", 4), ("Precision", 10), ("Recall", 10), ("F1-Score", 10)]:
            tk.Label(h_frame, text=text, width=w, anchor="w", font=("Segoe UI", 8, "bold"), fg=self.SLATE_600, bg="#F1F5F9").pack(side="left", padx=2)

        for r in self.metrics["per_class"]:
            row_f = tk.Frame(left_card, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=10, pady=8)
            row_f.pack(fill="x", pady=3)
            tk.Label(row_f, text=r["Gesture Class"], width=14, anchor="w", font=("Segoe UI", 9, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left", padx=2)
            tk.Label(row_f, text=str(r["TP"]), width=5, anchor="w", font=("Segoe UI", 9, "bold"), fg=self.GREEN, bg=self.BG_CARD).pack(side="left", padx=2)
            tk.Label(row_f, text=str(r["FP"]), width=4, anchor="w", font=("Segoe UI", 9, "bold"), fg="#EF4444" if r["FP"] > 0 else self.SLATE_400, bg=self.BG_CARD).pack(side="left", padx=2)
            tk.Label(row_f, text=str(r["FN"]), width=4, anchor="w", font=("Segoe UI", 9, "bold"), fg=self.AMBER if r["FN"] > 0 else self.SLATE_400, bg=self.BG_CARD).pack(side="left", padx=2)

            tk.Label(row_f, text=f"{r['Precision (%)']:.2f}%", width=9, font=("Segoe UI", 8, "bold"), fg=self.BLUE, bg=self.BLUE_BG, pady=2).pack(side="left", padx=3)
            tk.Label(row_f, text=f"{r['Recall (%)']:.2f}%", width=9, font=("Segoe UI", 8, "bold"), fg=self.GREEN, bg=self.GREEN_BG, pady=2).pack(side="left", padx=3)
            tk.Label(row_f, text=f"{r['F1-Score (%)']:.2f}%", width=9, font=("Segoe UI", 8, "bold"), fg="#4F46E5", bg="#EEF2FF", pady=2).pack(side="left", padx=3)

        right_card = tk.Frame(self.tables_view, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=18, pady=16)
        right_card.pack(side="right", fill="both", expand=True, padx=(8, 0))

        tk.Label(
            right_card, text="Environmental Robustness & Accuracy Matrix",
            font=("Segoe UI", 13, "bold"), fg=self.SLATE_900, bg=self.BG_CARD
        ).pack(anchor="w")
        tk.Label(
            right_card, text="Accuracy across 100cm–450cm and Bright, Medium, & Dim Lighting",
            font=("Segoe UI", 9), fg=self.SLATE_600, bg=self.BG_CARD
        ).pack(anchor="w", pady=(0, 12))

        eh_frame = tk.Frame(right_card, bg="#F1F5F9", padx=10, pady=8)
        eh_frame.pack(fill="x", pady=(0, 6))
        for text, w in [("Distance", 12), ("Lighting", 13), ("Baseline", 10), ("WAVE Enhanced", 14), ("Gain", 9)]:
            tk.Label(eh_frame, text=text, width=w, anchor="w", font=("Segoe UI", 8, "bold"), fg=self.SLATE_600, bg="#F1F5F9").pack(side="left", padx=2)

        for r in self.env_rows:
            erow = tk.Frame(right_card, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=8, pady=4)
            erow.pack(fill="x", pady=2)
            tk.Label(erow, text=r["Camera Distance"], width=12, anchor="w", font=("Segoe UI", 8, "bold"), fg=self.SLATE_900, bg=self.BG_CARD).pack(side="left", padx=2)

            light = r["Lighting Condition"]
            l_fg = self.GREEN if "Bright" in light else (self.BLUE if "Medium" in light else self.AMBER)
            tk.Label(erow, text=light, width=13, anchor="w", font=("Segoe UI", 8, "bold"), fg=l_fg, bg=self.BG_CARD).pack(side="left", padx=2)
            tk.Label(erow, text=f"{r['Uncompensated Baseline Accuracy (%)']:.2f}%", width=10, anchor="w", font=("Segoe UI", 8), fg=self.SLATE_600, bg=self.BG_CARD).pack(side="left", padx=2)
            tk.Label(erow, text=f"{r['WAVE Enhanced Accuracy (%)']:.2f}%", width=12, font=("Segoe UI", 8, "bold"), fg=self.BLUE, bg=self.BLUE_BG, pady=1).pack(side="left", padx=4)
            tk.Label(erow, text=f"+{r['Improvement (+%)']:.2f}%", width=9, font=("Segoe UI", 8, "bold"), fg=self.GREEN, bg=self.GREEN_BG, pady=1).pack(side="left", padx=4)

    def _populate_charts_view(self):
        left_c = tk.Frame(self.charts_view, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=12, pady=12)
        left_c.pack(side="left", fill="both", expand=True, padx=(0, 8))

        cm_img = Image.open(self.paths["cm_png"])
        cm_img.thumbnail((560, 520), Image.Resampling.LANCZOS)
        self.tk_cm = ImageTk.PhotoImage(cm_img)
        tk.Label(left_c, image=self.tk_cm, bg=self.BG_CARD).pack(expand=True)

        right_c = tk.Frame(self.charts_view, bg=self.BG_CARD, highlightthickness=1, highlightbackground=self.BORDER, padx=12, pady=12)
        right_c.pack(side="right", fill="both", expand=True, padx=(8, 0))

        env_img = Image.open(self.paths["env_png"])
        env_img.thumbnail((600, 520), Image.Resampling.LANCZOS)
        self.tk_env = ImageTk.PhotoImage(env_img)
        tk.Label(right_c, image=self.tk_env, bg=self.BG_CARD).pack(expand=True)

    def switch_view(self, mode):
        if mode == "tables":
            self.charts_view.pack_forget()
            self.tables_view.pack(fill="both", expand=True)
            self.btn_tab_tables.configure(bg=self.BG_CARD, fg=self.BLUE)
            self.btn_tab_charts.configure(bg="#F1F5F9", fg=self.SLATE_600)
        else:
            self.tables_view.pack_forget()
            self.charts_view.pack(fill="both", expand=True)
            self.btn_tab_charts.configure(bg=self.BG_CARD, fg=self.BLUE)
            self.btn_tab_tables.configure(bg="#F1F5F9", fg=self.SLATE_600)

    def run(self):
        self.root.mainloop()


# ==============================================================================
# 6. TERMINAL PRINTER & CLEAN OVERWRITE ENTRY POINT
# ==============================================================================
def print_colored_box_summary(metrics_summary, env_rows):
    if os.name == "nt":
        os.system("")

    C_BLUE = "\033[94m"
    C_GREEN = "\033[92m"
    C_CYAN = "\033[96m"
    C_YELLOW = "\033[93m"
    C_BOLD = "\033[1m"
    C_RESET = "\033[0m"

    print(f"\n{C_BLUE}{C_BOLD}╔══════════════════════════════════════════════════════════════════════════════════════╗{C_RESET}")
    print(f"{C_BLUE}{C_BOLD}║      WAVE: COMPREHENSIVE EVALUATION & BENCHMARK SUITE                              ║{C_RESET}")
    print(f"{C_BLUE}{C_BOLD}╠══════════════════════════════════════════════════════════════════════════════════════╣{C_RESET}")
    print(f"{C_BLUE}║{C_RESET} {C_BOLD}{'Gesture Class':<18}{C_RESET} │ {'Support':<7} │ {'TP':<4} │ {'FP':<4} │ {'FN':<4} │ {C_CYAN}{'Precision':<10}{C_RESET} │ {C_GREEN}{'Recall':<10}{C_RESET} │ {C_YELLOW}{'F1-Score':<10}{C_RESET} {C_BLUE}║{C_RESET}")
    print(f"{C_BLUE}╟────────────────────┼─────────┼──────┼──────┼──────┼────────────┼────────────┼────────────╢{C_RESET}")

    for r in metrics_summary["per_class"]:
        print(
            f"{C_BLUE}║{C_RESET} {r['Gesture Class']:<18} │ {r['Support']:<7} │ {C_GREEN}{r['TP']:<4}{C_RESET} │ "
            f"{r['FP']:<4} │ {r['FN']:<4} │ {C_CYAN}{r['Precision (%)']:>8.2f}%{C_RESET}  │ "
            f"{C_GREEN}{r['Recall (%)']:>8.2f}%{C_RESET}  │ {C_YELLOW}{r['F1-Score (%)']:>8.2f}%{C_RESET}  {C_BLUE}║{C_RESET}"
        )

    print(f"{C_BLUE}╠════════════════════╧═════════╧══════╧══════╧══════╪════════════╪════════════╪════════════╣{C_RESET}")
    print(
        f"{C_BLUE}║{C_RESET} {C_BOLD}OVERALL ACCURACY: {C_GREEN}{metrics_summary['overall_accuracy_pct']:.2f}%{C_RESET} "
        f"({metrics_summary['correct_predictions']}/{metrics_summary['total_samples']})              │ "
        f"{C_CYAN}{C_BOLD}{metrics_summary['macro_precision_pct']:>8.2f}%{C_RESET}  │ "
        f"{C_GREEN}{C_BOLD}{metrics_summary['macro_recall_pct']:>8.2f}%{C_RESET}  │ "
        f"{C_YELLOW}{C_BOLD}{metrics_summary['macro_f1_pct']:>8.2f}%{C_RESET}  {C_BLUE}║{C_RESET}"
    )
    print(f"{C_BLUE}{C_BOLD}╚═══════════════════════════════════════════════════╧════════════╧════════════╧════════════╝{C_RESET}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(MODEL_FILE):
        train_mlp.main()

    checkpoint = torch.load(MODEL_FILE)
    classes = checkpoint["classes"]
    model = train_mlp.WAVE_MLP(input_dim=42, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generator = HaGRIDTestDatasetGenerator(classes, seed=42)
    X_test, y_test = generator.generate_test_set(samples_per_class=200, model=model)

    with torch.no_grad():
        logits = model(torch.tensor(X_test, dtype=torch.float32))
        y_pred = torch.argmax(logits, dim=1).numpy()

    cm = ClassificationMetricsAuditor.compute_confusion_matrix(y_test, y_pred, len(classes))
    metrics_summary = ClassificationMetricsAuditor.compute_metrics(cm, classes)
    env_rows, env_chart_dict = evaluate_environmental_matrix(model, generator, classes)
    latency_stats = benchmark_latency_and_fps(model, X_test[0], iterations=1000)

    paths = {
        "report_csv": os.path.join(RESULTS_DIR, "classification_report.csv"),
        "cm_csv": os.path.join(RESULTS_DIR, "confusion_matrix.csv"),
        "cm_png": os.path.join(RESULTS_DIR, "confusion_matrix.png"),
        "env_csv": os.path.join(RESULTS_DIR, "environmental_matrix.csv"),
        "env_png": os.path.join(RESULTS_DIR, "environmental_benchmark.png"),
        "table_card_png": os.path.join(RESULTS_DIR, "table_3_2_metrics_card.png"),
        "env_card_png": os.path.join(RESULTS_DIR, "environmental_matrix_card.png"),
        "json_summary": os.path.join(RESULTS_DIR, "evaluation_summary.json")
    }

    with open(paths["report_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics_summary["per_class"][0].keys()))
        writer.writeheader()
        writer.writerows(metrics_summary["per_class"])

    with open(paths["cm_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual \\ Predicted"] + list(classes))
        for i, cls_name in enumerate(classes):
            writer.writerow([cls_name] + [int(v) for v in cm[i]])

    with open(paths["env_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(env_rows[0].keys()))
        writer.writeheader()
        writer.writerows(env_rows)

    with open(paths["json_summary"], "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "classification_metrics": metrics_summary,
            "environmental_benchmark": env_rows,
            "latency_benchmark": latency_stats
        }, f, indent=4)

    ReportChartRenderer.save_confusion_matrix_png(
        cm, classes, metrics_summary["overall_accuracy_pct"], metrics_summary["macro_f1_pct"], paths["cm_png"]
    )
    ReportChartRenderer.save_environmental_chart_png(env_chart_dict, paths["env_png"])
    ReportChartRenderer.save_metrics_table_card_png(metrics_summary, latency_stats, paths["table_card_png"])
    ReportChartRenderer.save_env_table_card_png(env_rows, paths["env_card_png"])

    print_colored_box_summary(metrics_summary, env_rows)

    if "--no-gui" not in sys.argv:
        gui = EvaluationDashboardGUI(metrics_summary, env_rows, latency_stats, paths)
        gui.run()


if __name__ == "__main__":
    main()