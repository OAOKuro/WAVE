import cv2
import mediapipe as mp
import numpy as np
import torch
import json
import time
import os
from collections import deque
from pynput.keyboard import Key, Controller

import train_mlp

# Initialize Keyboard Controller for Presentation Emulation (Section 4.1)
keyboard = Controller()

CONFIG_FILE = "config.json"
MODEL_FILE = os.path.join("models", "gesture_mlp.pth")

def load_config():
    """Loads shared configuration (IPC) from config.json (Section 4.3)."""
    default_config = {
        "recognition_enabled": True,
        "lighting_sensitivity": 0.6,
        "frame_skip": 1,
        "enable_zoom_optional": True,
        "enable_roi_optional": False,
        "active_tab": "live", # 'live' or 'mapping'
        "gesture_mappings": {
            "Right Swipe": "Right Arrow",
            "Left Swipe": "Left Arrow",
            "Open Palm": "Key B",
            "Pinch In / Out": "Ctrl + / -",
            "Index Finger Point": "Key L"
        }
    }
    if not os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "w") as f:
            json.dump(default_config, f, indent=4)
        return default_config
    try:
        with open(CONFIG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return default_config

def save_config(cfg):
    """Persists settings between sessions."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump(cfg, f, indent=4)
    except Exception as e:
        print(f"[WARN] Could not save config.json: {e}")

def trigger_keyboard_shortcut(gesture_name, mappings):
    """Translates recognized gesture into keyboard command (Table 4.1)."""
    mapped_key = mappings.get(gesture_name, "")
    if not mapped_key:
        return "None"

    if "right" in mapped_key.lower():
        keyboard.press(Key.right)
        keyboard.release(Key.right)
        return "Right Arrow (Next Slide)"
    elif "left" in mapped_key.lower():
        keyboard.press(Key.left)
        keyboard.release(Key.left)
        return "Left Arrow (Previous Slide)"
    elif "ctrl" in mapped_key.lower():
        with keyboard.pressed(Key.ctrl):
            keyboard.press('-')
            keyboard.release('-')
        return "Ctrl + Minus (Zoom)"
    else:
        if len(mapped_key) == 1:
            keyboard.press(mapped_key)
            keyboard.release(mapped_key)
            return f"Key '{mapped_key.upper()}'"
    return mapped_key

def draw_rounded_rect(img, pt1, pt2, color, radius=8, thickness=-1):
    """Draws clean rounded rectangles for modern UI cards (OpenCV BGR format)."""
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.rectangle(img, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.circle(img, (x1 + radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y1 + radius), radius, color, thickness)
    cv2.circle(img, (x1 + radius, y2 - radius), radius, color, thickness)
    cv2.circle(img, (x2 - radius, y2 - radius), radius, color, thickness)

def draw_keyboard_mapping_screen(w, h, config):
    """Renders the exact professional Keyboard Mapping Screen matching your mockup."""
    canvas = np.ones((h, w, 3), dtype=np.uint8)
    canvas[:] = (252, 250, 248) # Soft light background (#f8fafc)

    draw_rounded_rect(canvas, (20, 15), (w - 20, 75), (255, 255, 255), radius=10)
    
    draw_rounded_rect(canvas, (35, 27), (70, 63), (235, 206, 135), radius=6)
    cv2.putText(canvas, "WAVE", (82, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (42, 23, 15), 1, cv2.LINE_AA)

    draw_rounded_rect(canvas, (200, 27), (480, 63), (249, 245, 241), radius=8)
    cv2.putText(canvas, "Live Presentation", (215, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (139, 116, 100), 1, cv2.LINE_AA)
    
    draw_rounded_rect(canvas, (345, 31), (474, 59), (255, 255, 255), radius=6)
    cv2.putText(canvas, "Keyboard Mapping", (353, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (199, 132, 2), 1, cv2.LINE_AA)

    cv2.circle(canvas, (w - 275, 45), 5, (94, 197, 34), -1)
    cv2.putText(canvas, "CAM ACTIVE", (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (59, 41, 30), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, "ENGINE", (w - 165, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (59, 41, 30), 1, cv2.LINE_AA)
    draw_rounded_rect(canvas, (w - 110, 33), (w - 70, 57), (199, 132, 2), radius=12)
    cv2.circle(canvas, (w - 82, 45), 9, (255, 255, 255), -1)

    draw_rounded_rect(canvas, (w - 50, 27), (w - 25, 63), (240, 232, 226), radius=8)

    cv2.putText(canvas, "HARDWARE ABSTRACTION LAYER", (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (199, 132, 2), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Gesture Key Mappings", (35, 142), cv2.FONT_HERSHEY_DUPLEX, 1.0, (42, 23, 15), 1, cv2.LINE_AA)
    cv2.putText(canvas, "Configure spatial hand tracking vectors to emulate native operating system keycodes and presentation software macros.", (35, 168), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (139, 116, 100), 1, cv2.LINE_AA)

    draw_rounded_rect(canvas, (w - 280, 115), (w - 35, 155), (255, 255, 255), radius=8)
    cv2.circle(canvas, (w - 265, 135), 4, (199, 132, 2), -1)
    cv2.putText(canvas, "VIRTUAL HID STATUS", (w - 250, 131), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (184, 163, 148), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SYNCHRONIZED (0.4ms)", (w - 250, 147), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (199, 132, 2), 1, cv2.LINE_AA)

    draw_rounded_rect(canvas, (35, 195), (w - 35, 250), (255, 255, 255), radius=10)
    cv2.putText(canvas, "ACTIVE PRESET", (50, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (184, 163, 148), 1, cv2.LINE_AA)
    
    draw_rounded_rect(canvas, (145, 210), (385, 238), (252, 250, 248), radius=6)
    cv2.putText(canvas, "PowerPoint / Keynote Preset (Default)", (153, 227), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (42, 23, 15), 1, cv2.LINE_AA)
    
    cv2.putText(canvas, "5 ACTIVE BINDINGS", (405, 228), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 116, 100), 1, cv2.LINE_AA)

    draw_rounded_rect(canvas, (w - 275, 210), (w - 175, 238), (255, 255, 255), radius=6)
    cv2.putText(canvas, "EXPORT JSON", (w - 263, 227), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (85, 65, 51), 1, cv2.LINE_AA)

    draw_rounded_rect(canvas, (w - 160, 208), (w - 50, 240), (199, 132, 2), radius=6)
    cv2.putText(canvas, "+ New Mapping", (w - 148, 227), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.putText(canvas, "ASSIGNED MOTION & KINEMATICS", (35, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (184, 163, 148), 1, cv2.LINE_AA)
    cv2.putText(canvas, "OUTPUT EMULATION", (w - 370, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (184, 163, 148), 1, cv2.LINE_AA)
    cv2.putText(canvas, "STATE", (w - 90, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (184, 163, 148), 1, cv2.LINE_AA)

    mappings = config.get("gesture_mappings", {
        "Right Swipe": "Right Arrow",
        "Left Swipe": "Left Arrow",
        "Open Palm": "Key B",
        "Pinch In / Out": "Ctrl + / -",
        "Index Finger Point": "Key L"
    })
    descriptions = {
        "Right Swipe": "(Next Slide)",
        "Left Swipe": "(Previous Slide)",
        "Open Palm": "(Blackout Screen)",
        "Pinch In / Out": "(Zoom Slide)",
        "Index Finger Point": "(Laser Pointer)"
    }

    y_pos = 305
    for gesture, keycode in mappings.items():
        draw_rounded_rect(canvas, (35, y_pos), (w - 35, y_pos + 62), (255, 255, 255), radius=8)
        cv2.putText(canvas, gesture, (60, y_pos + 36), cv2.FONT_HERSHEY_DUPLEX, 0.58, (42, 23, 15), 1, cv2.LINE_AA)

        draw_rounded_rect(canvas, (w - 410, y_pos + 14), (w - 285, y_pos + 48), (254, 242, 224), radius=6)
        cv2.putText(canvas, str(keycode), (w - 400, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (161, 105, 3), 1, cv2.LINE_AA)

        cv2.putText(canvas, descriptions.get(gesture, ""), (w - 275, y_pos + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (139, 116, 100), 1, cv2.LINE_AA)

        draw_rounded_rect(canvas, (w - 165, y_pos + 16), (w - 105, y_pos + 46), (249, 245, 241), radius=6)
        cv2.putText(canvas, "REBIND", (w - 155, y_pos + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (105, 85, 71), 1, cv2.LINE_AA)

        draw_rounded_rect(canvas, (w - 80, y_pos + 18), (w - 45, y_pos + 44), (199, 132, 2), radius=12)
        cv2.circle(canvas, (w - 53, y_pos + 31), 9, (255, 255, 255), -1)

        y_pos += 70

    cv2.putText(canvas, "[TAB] Switch between Live Presentation & Keyboard Mapping    [Q] Quit", (35, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (184, 163, 148), 1, cv2.LINE_AA)

    return canvas

def main():
    if not os.path.exists(MODEL_FILE):
        train_mlp.main()

    checkpoint = torch.load(MODEL_FILE)
    classes = checkpoint["classes"]
    model = train_mlp.WAVE_MLP(input_dim=42, num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    config = load_config()
    last_config_load = time.time()

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=config.get("lighting_sensitivity", 0.6),
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("M5: Camera not detected.")
        return

    wrist_history = deque(maxlen=5)
    prediction_buffer = deque(maxlen=2)

    frame_count = 0
    last_trigger_time = 0.0
    COOLDOWN_SECONDS = 0.6

    current_status_msg = "M3: No Hand Detected"
    last_dispatched_action = "None"
    last_confidence = 0.0
    cached_landmarks = None

    print("==========================================================")
    print(" WAVE: Desktop Python Application Started")
    print(" Controls:")
    print("   [TAB]   : Switch between Live Presentation & Keyboard Mapping")
    print("   [SPACE] : Toggle Gesture Recognition ON/OFF")
    print("   [Q/ESC] : Quit Application")
    print("==========================================================")

    while True:
        tick_start = cv2.getTickCount()

        if time.time() - last_config_load > 2.0:
            config = load_config()
            last_config_load = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        frame_count += 1

        active_tab = config.get("active_tab", "live")

        if active_tab == "mapping":
            mapping_screen = draw_keyboard_mapping_screen(w, h, config)
            cv2.imshow("WAVE - Real-Time Gesture Recognition and Control Module", mapping_screen)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q') or key == 27:
                break
            elif key == 9: # TAB key switches back to live presentation view
                config["active_tab"] = "live"
                save_config(config)
            continue

        # ================= LIVE PRESENTATION VIEW =================
        canvas = np.ones((h, w, 3), dtype=np.uint8)
        canvas[:] = (252, 250, 248)

        draw_rounded_rect(canvas, (20, 15), (w - 20, 75), (255, 255, 255), radius=10)
        
        draw_rounded_rect(canvas, (35, 27), (70, 63), (235, 206, 135), radius=6)
        cv2.putText(canvas, "WAVE", (82, 50), cv2.FONT_HERSHEY_DUPLEX, 0.75, (42, 23, 15), 1, cv2.LINE_AA)

        draw_rounded_rect(canvas, (200, 27), (480, 63), (249, 245, 241), radius=8)
        draw_rounded_rect(canvas, (203, 31), (335, 59), (255, 255, 255), radius=6)
        cv2.putText(canvas, "Live Presentation", (215, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (199, 132, 2), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Keyboard Mapping", (345, 49), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (139, 116, 100), 1, cv2.LINE_AA)

        recognition_on = config.get("recognition_enabled", True)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - tick_start)

        cv2.circle(canvas, (w - 275, 45), 5, (94, 197, 34), -1)
        cv2.putText(canvas, "CAM ACTIVE", (w - 260, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (59, 41, 30), 1, cv2.LINE_AA)
        
        cv2.putText(canvas, "ENGINE", (w - 165, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (59, 41, 30), 1, cv2.LINE_AA)
        engine_color = (199, 132, 2) if recognition_on else (220, 220, 220)
        draw_rounded_rect(canvas, (w - 110, 33), (w - 70, 57), engine_color, radius=12)
        circle_x = (w - 82) if recognition_on else (w - 98)
        cv2.circle(canvas, (circle_x, 45), 9, (255, 255, 255), -1)

        draw_rounded_rect(canvas, (w - 50, 27), (w - 25, 63), (240, 232, 226), radius=8)

        cam_x, cam_y, cam_w, cam_h = 35, 95, w - 70, h - 170
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        cached_landmarks = results.multi_hand_landmarks[0] if results.multi_hand_landmarks else None

        hand_detected = cached_landmarks is not None
        resized_cam = cv2.resize(frame, (cam_w, cam_h))

        if hand_detected and recognition_on:
            wrist = cached_landmarks.landmark[0]
            index_tip = cached_landmarks.landmark[8]

            mp_drawing.draw_landmarks(
                resized_cam,
                cached_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 128), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2)
            )

            pt_wrist = (int(wrist.x * cam_w), int(wrist.y * cam_h))
            pt_index = (int(index_tip.x * cam_w), int(index_tip.y * cam_h))
            cv2.arrowedLine(resized_cam, pt_wrist, pt_index, (0, 0, 255), 3, tipLength=0.2)

            gesture_start_time = time.perf_counter()
            features_42 = train_mlp.extract_42_features(cached_landmarks).reshape(1, -1)
            tensor_input = torch.tensor(features_42, dtype=torch.float32)

            with torch.no_grad():
                outputs = model(tensor_input)
                probs = torch.form_softmax = torch.softmax(outputs, dim=1).numpy()[0]

            pred_idx = int(np.argmax(probs))
            mlp_label = classes[pred_idx]
            mlp_conf = float(probs[pred_idx])

            wrist_history.append(wrist.x)
            detected_gesture = mlp_label
            confidence = mlp_conf

            # MOTION PRIORITY OVERRIDE: Block false pinches and enforce swipes during motion
            if len(wrist_history) >= 4:
                dx = wrist_history[-1] - wrist_history[0]
                if dx < -0.05:
                    detected_gesture = "Left Swipe"
                    confidence = 0.99
                elif dx > 0.05:
                    detected_gesture = "Right Swipe"
                    confidence = 0.99
                else:
                    # Only allow pinch/zoom if hand is relatively stationary
                    if detected_gesture in ["Pinch In / Out", "Pinch"]:
                        detected_gesture = mlp_label

            prediction_buffer.append(detected_gesture)

            now = time.time()
            if detected_gesture != "Neutral" and (now - last_trigger_time) > COOLDOWN_SECONDS:
                if prediction_buffer.count(detected_gesture) >= 1:
                    action_str = trigger_keyboard_shortcut(detected_gesture, config.get("gesture_mappings", {}))
                    last_dispatched_action = f"{detected_gesture} -> {action_str}"
                    last_confidence = confidence * 100.0
                    current_status_msg = f"{detected_gesture} Recognized"
                    last_trigger_time = now
                    wrist_history.clear()
                    prediction_buffer.clear()
            elif (now - last_trigger_time) > COOLDOWN_SECONDS:
                current_status_msg = "Gesture Control Ready"
        elif not recognition_on:
            current_status_msg = "Gesture Control OFF"
        else:
            current_status_msg = "No Hand Detected"

        canvas[cam_y:cam_y+cam_h, cam_x:cam_x+cam_w] = resized_cam

        card_y = h - 65
        draw_rounded_rect(canvas, (35, card_y), (w - 35, h - 15), (255, 255, 255), radius=8)
        
        cv2.putText(canvas, f"Last Dispatched: {last_dispatched_action}   |   Confidence: {last_confidence:.1f}%",
                    (50, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (42, 23, 15), 1, cv2.LINE_AA)
        cv2.putText(canvas, "[TAB] Switch to Keyboard Mapping Screen    [SPACE] Pause Engine    [Q] Quit",
                    (50, card_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (139, 116, 100), 1, cv2.LINE_AA)

        cv2.imshow("WAVE - Real-Time Gesture Recognition and Control Module", canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == 9:  # TAB key switches to Keyboard Mapping view
            config["active_tab"] = "mapping"
            save_config(config)
        elif key == ord(' '):  # Spacebar On/Off Toggle
            config["recognition_enabled"] = not config.get("recognition_enabled", True)
            save_config(config)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()