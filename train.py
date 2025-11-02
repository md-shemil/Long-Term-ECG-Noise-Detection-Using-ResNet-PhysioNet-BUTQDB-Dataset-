# Install needed libraries first (you only do this once):
# pip install pandas wfdb torch scikit-learn

import os
import pandas as pd
import wfdb
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Subset
from sklearn.model_selection import KFold

def augment(segment):
    noise = np.random.normal(0, 0.01, segment.shape)
    segment = segment + noise
    scale = np.random.uniform(0.85, 1.15)
    segment = segment * scale
    shift = int(np.random.uniform(-50, 50))
    if shift > 0:
        segment = np.pad(segment, (shift, 0))[:len(segment)]
    elif shift < 0:
        segment = np.pad(segment, (0, -shift))[(-shift):]
    return segment

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

class ResNetECG(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 64, 7, 2, 3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(3, 2, 1)
        )
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.layer3 = ResidualBlock(64)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

DATA_ROOT = 'dataset'
subject_folders = [f for f in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, f))]

all_segments = []
all_labels_binary = []

clean_code = 1
noisy_code = 2

for subj in subject_folders:
    ann_path = os.path.join(DATA_ROOT, subj, f'{subj}_ANN.csv')
    ecg_path = os.path.join(DATA_ROOT, subj, f'{subj}_ECG')

    if not os.path.exists(ann_path) or not os.path.exists(ecg_path + '.dat'):
        continue

    ann = pd.read_csv(ann_path, header=None)
    record = wfdb.rdrecord(ecg_path)
    ecg_signal = record.p_signal

    if ecg_signal.ndim > 1:
        ecg_signal = ecg_signal[:, 0]

    for i, row in ann.iterrows():
        if pd.isna(row[0]) or pd.isna(row[1]) or pd.isna(row[2]):
            continue

        start, end = int(row[0]), int(row[1])
        label_code = int(row[2])

        if label_code == clean_code:
            label_bin = 0
        elif label_code == noisy_code:
            label_bin = 1
        else:
            continue

        if end - start < 250:
            continue

        segment = ecg_signal[start:end]
        segment = (segment - np.mean(segment)) / np.std(segment)

        desired_len = 2500
        if len(segment) < desired_len:
            segment = np.pad(segment, (0, desired_len - len(segment)))
        elif len(segment) > desired_len:
            segment = segment[:desired_len]

        all_segments.append(segment)
        all_labels_binary.append(label_bin)

X = torch.tensor(np.array(all_segments), dtype=torch.float32).unsqueeze(1)
y = torch.tensor(all_labels_binary, dtype=torch.long)

dataset = TensorDataset(X, y)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 40
patience = 7
best_val_accuracy_overall = 0

for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
    print(f"Fold {fold + 1}")
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    model = ResNetECG()
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    loss_fn = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=10, T_mult=2)

    best_fold_val_acc = 0
    early_stop_counter = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb_np = xb.squeeze(1).numpy()
            xb_aug = np.array([augment(x) for x in xb_np])
            xb_aug = torch.tensor(xb_aug, dtype=torch.float32).unsqueeze(1)

            preds = model(xb_aug)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()
            optim.zero_grad()
        scheduler.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                preds = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total += len(yb)
        val_accuracy = correct / total
        print(f"Epoch {epoch + 1}, Val Accuracy: {val_accuracy:.4f}")

        if val_accuracy > best_fold_val_acc:
            best_fold_val_acc = val_accuracy
            # Save best model of this fold
            torch.save(model.state_dict(), f'trybest_model_fold-{fold + 1}.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    if best_fold_val_acc > best_val_accuracy_overall:
        best_val_accuracy_overall = best_fold_val_acc

print(f"Best cross-validation accuracy: {best_val_accuracy_overall:.4f}")
