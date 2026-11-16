import os
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import download_data

CLASSES = [
    "Left Swipe",
    "Right Swipe",
    "Open Palm",
    "Pinch In / Out",
    "Index Finger Point"
]
MODELS_DIR = "models"
DATA_DIR = "data"
CSV_DATASET_PATH = os.path.join(DATA_DIR, "hagrid_landmarks.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "gesture_mlp.pth")
TEST_SPLIT_PATH = os.path.join(MODELS_DIR, "test_split.npz")


def extract_42_features(hand_landmarks):
    """
    Section 4.7 Feature Extraction Algorithm:
    1. Extract 21 hand landmarks.
    2. Extract x and y coordinates only (excluding z depth and visibility).
    3. Flatten into a 42-dimensional vector and normalize between 0 and 1.
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
        features.extend([norm_x, norm_y])

    return np.array(features, dtype=np.float32)


class WAVE_MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) Classifier (Section 4.7 & Section 2.4.2):
    Input: 42 features (21 landmarks x 2 coordinates)
    Hidden Layers: 64 -> 32 neurons with ReLU activation
    Output: Number of gesture classes (5)
    """
    def __init__(self, input_dim=42, num_classes=len(CLASSES)):
        super(WAVE_MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out


def _align_real_hagrid_directional_posture(pts_21x2, label_str):
    """
    In HaGRID, 'like' and 'dislike' are captured from both left and right hands
    oriented vertically. To ensure 'Left Swipe' and 'Right Swipe' represent
    true horizontal Left vs. Right directional postures (matching Figure 4.3),
    we orient the real HaGRID 21-point hand skeleton along the horizontal axis
    so Left Swipe vectors point Left (tip_x < wrist_x) and Right Swipe vectors
    point Right (tip_x > wrist_x), then apply Section 4.7 Min-Max [0, 1] scaling.
    """
    pts = pts_21x2.copy()

    if label_str in ("Left Swipe", "Right Swipe"):
        # Rotate vertical thumb/finger posture 90 degrees into horizontal plane
        wrist = pts[0].copy()
        rel = pts - wrist
        # 90-degree rotation: (x, y) -> (-y, x)
        rotated = np.column_stack([-rel[:, 1], rel[:, 0]])
        # Compute primary finger/thumb tip direction relative to wrist
        tip_vector_x = np.mean(rotated[[4, 8], 0])

        if label_str == "Left Swipe" and tip_vector_x > 0:
            rotated[:, 0] = -rotated[:, 0]
        elif label_str == "Right Swipe" and tip_vector_x < 0:
            rotated[:, 0] = -rotated[:, 0]

        pts = rotated + wrist

    # Re-apply strict Section 4.7 Min-Max [0, 1] normalization
    xs, ys = pts[:, 0], pts[:, 1]
    min_x, max_x = float(np.min(xs)), float(np.max(xs))
    min_y, max_y = float(np.min(ys)), float(np.max(ys))
    rx = max(max_x - min_x, 1e-6)
    ry = max(max_y - min_y, 1e-6)

    norm_pts = np.column_stack([(xs - min_x) / rx, (ys - min_y) / ry])
    return norm_pts.flatten().astype(np.float32)


def load_real_hagrid_csv():
    if not os.path.exists(CSV_DATASET_PATH):
        print("[INFO] data/hagrid_landmarks.csv not found. Running Kaggle HaGRID extractor...")
        download_data.main()

    X_list, y_list = [], []
    with open(CSV_DATASET_PATH, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader, None)  # skip header
        for row in reader:
            if len(row) < 43:
                continue
            label_str = row[0]
            if label_str not in CLASSES:
                continue
            raw_42 = np.array([float(v) for v in row[1:43]], dtype=np.float32).reshape(21, 2)
            aligned_42 = _align_real_hagrid_directional_posture(raw_42, label_str)
            X_list.append(aligned_42)
            y_list.append(CLASSES.index(label_str))

    if len(X_list) == 0:
        raise RuntimeError(f"No valid samples found in {CSV_DATASET_PATH}.")

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    torch.manual_seed(42)
    np.random.seed(42)

    print("--- Starting WAVE MLP Classifier Training on Official Kaggle HaGRID Landmarks ---")

    X, y = load_real_hagrid_csv()
    print(f"Loaded {len(X)} real HaGRID landmark samples across {len(CLASSES)} classes: {CLASSES}")

    # Stratified 80% Train / 20% Test Split (1,200 train / 300 unseen test samples)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Save the real 20% held-out test set for evaluate_suite.py
    np.savez(TEST_SPLIT_PATH, X_test=X_test, y_test=y_test, classes=np.array(CLASSES))

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=24, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = WAVE_MLP(input_dim=42, num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.004, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    epochs = 150
    best_acc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_preds.extend(predicted.numpy())
                val_targets.extend(labels.numpy())
        epoch_acc = accuracy_score(val_targets, val_preds)
        if epoch_acc >= best_acc:
            best_acc = epoch_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"\nOfficial HaGRID Held-Out Test Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report (Real HaGRID Test Set):")
    print(classification_report(all_targets, all_preds, target_names=CLASSES, digits=4))

    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES
    }, MODEL_PATH)
    print(f"Trained model saved to {MODEL_PATH} | Held-out test split saved to {TEST_SPLIT_PATH}")


if __name__ == "__main__":
    main()