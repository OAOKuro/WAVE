import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np

CLASSES = ["Left Swipe", "Right Swipe", "Pinch In", "Pinch Out"]
MODELS_DIR = "models"

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
    Multi-Layer Perceptron (MLP) Classifier (Section 4.7 & Section 2.4.2)[cite: 40, 104]:
    Input: 42 features (21 landmarks x 2 coordinates)[cite: 13, 104]
    Hidden Layers: 64 -> 32 neurons with ReLU activation[cite: 104]
    Output: Number of gesture classes[cite: 104]
    """
    def __init__(self, input_dim=42, num_classes=4):
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

def generate_synthetic_training_data(samples_per_class=1200):
    """
    Generates structured 42-feature landmark tensors representing hand gestures 
    under diverse orientations and scales (simulating HaGRID landmark distribution).
    """
    X, y = [], []
    np.random.seed(42)

    for label_idx, class_name in enumerate(CLASSES):
        for _ in range(samples_per_class):
            features = np.zeros(42, dtype=np.float32)
            features[0] = 0.5 + np.random.normal(0, 0.05)
            features[1] = 0.9 + np.random.normal(0, 0.02)

            if class_name == "Left Swipe":
                for i in range(1, 21):
                    features[2*i] = np.clip(0.25 + (i % 4) * 0.05 + np.random.normal(0, 0.04), 0.0, 1.0)
                    features[2*i+1] = np.clip(0.4 + (i // 5) * 0.1 + np.random.normal(0, 0.04), 0.0, 1.0)
            elif class_name == "Right Swipe":
                for i in range(1, 21):
                    features[2*i] = np.clip(0.65 + (i % 4) * 0.05 + np.random.normal(0, 0.04), 0.0, 1.0)
                    features[2*i+1] = np.clip(0.4 + (i // 5) * 0.1 + np.random.normal(0, 0.04), 0.0, 1.0)
            elif class_name == "Pinch In":
                for i in range(1, 21):
                    features[2*i] = np.clip(0.5 + np.random.normal(0, 0.06), 0.0, 1.0)
                    features[2*i+1] = np.clip(0.5 + np.random.normal(0, 0.06), 0.0, 1.0)
                features[8], features[9] = 0.49 + np.random.normal(0, 0.01), 0.49 + np.random.normal(0, 0.01)
                features[16], features[17] = 0.51 + np.random.normal(0, 0.01), 0.51 + np.random.normal(0, 0.01)
            elif class_name == "Pinch Out":
                for i in range(1, 21):
                    features[2*i] = np.clip(0.5 + np.random.normal(0, 0.1), 0.0, 1.0)
                    features[2*i+1] = np.clip(0.5 + np.random.normal(0, 0.1), 0.0, 1.0)
                features[8], features[9] = 0.15 + np.random.normal(0, 0.02), 0.3 + np.random.normal(0, 0.02)
                features[16], features[17] = 0.85 + np.random.normal(0, 0.02), 0.3 + np.random.normal(0, 0.02)

            X.append(features)
            y.append(label_idx)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print("--- Starting WAVE MLP Classifier Training ---")
    
    X, y = generate_synthetic_training_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = WAVE_MLP(input_dim=42, num_classes=len(CLASSES))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 15
    for epoch in range(epochs):
        model.train()
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.numpy())
            all_targets.extend(labels.numpy())

    acc = accuracy_score(all_targets, all_preds)
    print(f"Model Retrained. Test Accuracy: {acc * 100:.2f}%")

    model_path = os.path.join(MODELS_DIR, "gesture_mlp.pth")
    torch.save({
        "model_state_dict": model.state_dict(),
        "classes": CLASSES
    }, model_path)
    print(f"Model successfully saved to {model_path}!")

if __name__ == "__main__":
    main()