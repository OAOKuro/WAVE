import os
import json
import csv
import glob
import zipfile
import cv2
import mediapipe as mp
import numpy as np

# 1. Configure Kaggle Authentication
kaggle_dir = os.path.expanduser("~/.kaggle")
os.makedirs(kaggle_dir, exist_ok=True)

KAGGLE_TOKEN = os.environ.get("KAGGLE_KEY", "KGAT_89e42b205cd9a67300c9a9e0809c93a7")
os.environ["KAGGLE_USERNAME"] = "natalie"
os.environ["KAGGLE_KEY"] = KAGGLE_TOKEN

token_file = os.path.join(kaggle_dir, "access_token")
with open(token_file, "w") as f:
    f.write(KAGGLE_TOKEN)

json_file = os.path.join(kaggle_dir, "kaggle.json")
with open(json_file, "w") as f:
    json.dump({"username": "natalie", "key": KAGGLE_TOKEN}, f)

from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = "./data"
CSV_OUTPUT = os.path.join(DATA_DIR, "hagrid_landmarks.csv")
DATASET_SLUG = "innominate817/hagrid-sample-120k-384p"

# Map official HaGRID gesture folders to WAVE's 5 presentation gestures
HAGRID_FOLDER_TO_WAVE_CLASS = {
    "dislike": "Left Swipe",
    "like": "Right Swipe",
    "palm": "Open Palm",
    "stop": "Open Palm",
    "ok": "Pinch In / Out",
    "one": "Index Finger Point"
}

SAMPLES_PER_CLASS = 300  # 300 real HaGRID images per class = 1,500 total samples


def extract_42_features(hand_landmarks):
    """
    Extracts 21 hand landmarks (x, y only -> 42 features) and normalizes between 0 and 1.
    """
    xs = [lm.x for lm in hand_landmarks.landmark]
    ys = [lm.y for lm in hand_landmarks.landmark]

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    range_x = (max_x - min_x) if (max_x - min_x) > 1e-6 else 1.0
    range_y = (max_y - min_y) if (max_y - min_y) > 1e-6 else 1.0

    features = []
    for lm in hand_landmarks.landmark:
        norm_x = (lm.x - min_x) / range_x
        norm_y = (lm.y - min_y) / range_y
        features.extend([round(float(norm_x), 6), round(float(norm_y), 6)])

    return features


def extract_from_zip_without_unpacking(zip_path, hands_detector):
    """
    Streams images directly from the Kaggle .zip archive in memory, extracts 42
    MediaPipe landmarks, and deletes the zip afterward to keep storage free.
    """
    class_counts = {cls_name: 0 for cls_name in set(HAGRID_FOLDER_TO_WAVE_CLASS.values())}
    rows = []

    print(f"[INFO] Streaming images directly from {zip_path} in memory...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        all_members = zf.namelist()
        for member in all_members:
            if not member.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            parts = member.replace("\\", "/").lower().split("/")
            matched_wave_class = None
            for folder_key, wave_cls in HAGRID_FOLDER_TO_WAVE_CLASS.items():
                if any(folder_key.lower() == p or f"_{folder_key.lower()}" in p for p in parts[:-1]):
                    matched_wave_class = wave_cls
                    break

            if matched_wave_class is None:
                continue

            if class_counts[matched_wave_class] >= SAMPLES_PER_CLASS:
                if all(c >= SAMPLES_PER_CLASS for c in class_counts.values()):
                    break
                continue

            try:
                img_bytes = zf.read(member)
                np_arr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    continue

                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                res = hands_detector.process(rgb)
                if res.multi_hand_landmarks:
                    feat_42 = extract_42_features(res.multi_hand_landmarks[0])
                    rows.append([matched_wave_class] + feat_42)
                    class_counts[matched_wave_class] += 1
                    if class_counts[matched_wave_class] % 50 == 0:
                        print(f"  -> {matched_wave_class}: {class_counts[matched_wave_class]}/{SAMPLES_PER_CLASS} extracted")
            except Exception:
                continue

    return rows, class_counts


def extract_from_unpacked_folders(hands_detector):
    class_counts = {cls_name: 0 for cls_name in set(HAGRID_FOLDER_TO_WAVE_CLASS.values())}
    rows = []

    for folder_key, wave_cls in HAGRID_FOLDER_TO_WAVE_CLASS.items():
        if class_counts[wave_cls] >= SAMPLES_PER_CLASS:
            continue
        pattern = os.path.join(DATA_DIR, "**", f"*{folder_key}*", "*.*")
        img_files = [
            p for p in glob.glob(pattern, recursive=True)
            if p.lower().endswith((".jpg", ".jpeg", ".png"))
        ]
        for img_path in img_files:
            if class_counts[wave_cls] >= SAMPLES_PER_CLASS:
                break
            img = cv2.imread(img_path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = hands_detector.process(rgb)
            if res.multi_hand_landmarks:
                feat_42 = extract_42_features(res.multi_hand_landmarks[0])
                rows.append([wave_cls] + feat_42)
                class_counts[wave_cls] += 1
                if class_counts[wave_cls] % 50 == 0:
                    print(f"  -> {wave_cls}: {class_counts[wave_cls]}/{SAMPLES_PER_CLASS} extracted")

    return rows, class_counts


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    api = KaggleApi()
    api.authenticate()
    print("[1/3] Kaggle API successfully authenticated!")

    zip_path = os.path.join(DATA_DIR, "hagrid-sample-120k-384p.zip")

    if not os.path.exists(zip_path) and not glob.glob(os.path.join(DATA_DIR, "**", "*.jpg"), recursive=True):
        print(f"[2/3] Downloading official Kaggle HaGRID archive ({DATASET_SLUG})...")
        api.dataset_download_files(DATASET_SLUG, path=DATA_DIR, unzip=False, quiet=False)

    mp_hands = mp.solutions.hands
    hands_detector = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.7
    )

    print("[3/3] Extracting 21-point (42-feature) MediaPipe landmarks from real HaGRID images...")
    if os.path.exists(zip_path):
        rows, counts = extract_from_zip_without_unpacking(zip_path, hands_detector)
        try:
            os.remove(zip_path)
            print(f"[CLEANUP] Automatically deleted {zip_path} to free laptop storage.")
        except Exception:
            pass
    else:
        rows, counts = extract_from_unpacked_folders(hands_detector)

    hands_detector.close()

    if not rows:
        raise RuntimeError("No landmarks were extracted. Please verify your ./data folder or Kaggle download.")

    header = ["gesture_label"] + [f"f_{i}" for i in range(42)]
    with open(CSV_OUTPUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print("\n==================================================================")
    print(" OFFICIAL KAGGLE HAGRID LANDMARK EXTRACTION COMPLETE!")
    print(f" Saved {len(rows)} real 42-feature vectors to: {CSV_OUTPUT}")
    print(f" Class Breakdown: {counts}")
    print("==================================================================")


if __name__ == "__main__":
    main()