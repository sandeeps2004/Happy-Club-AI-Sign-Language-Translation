"""
Training Script — Train LSTM classifier on INCLUDE-50 keypoints.

Usage:
    python train.py                     # Train with default settings
    python train.py --epochs 100        # Custom epochs
    python train.py --device cuda       # Force GPU
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import config
from model import SignLanguageLSTM
from preprocessing import normalize_sequence, add_velocity_features, augment_sequence


# ── Dataset ────────────────────────────────────────────────────────────

class KeypointDataset(Dataset):
    """Load pre-extracted keypoint .npy files with normalization + augmentation."""

    def __init__(self, root_dir, split="train", glosses=None, augment=False):
        self.samples = []
        self.labels = []
        self.augment = augment
        split_dir = Path(root_dir) / split

        if not split_dir.exists():
            raise FileNotFoundError(
                f"Split directory not found: {split_dir}\n"
                f"Run: python setup_dataset.py --sample  (for test data)\n"
                f"  or: python setup_dataset.py --download --extract  (for real data)"
            )

        # Discover classes from directory structure
        class_dirs = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
        if glosses is not None:
            self.glosses = glosses
        else:
            self.glosses = class_dirs

        self.gloss_to_idx = {g: i for i, g in enumerate(self.glosses)}

        for class_name in class_dirs:
            if class_name not in self.gloss_to_idx:
                continue
            class_dir = split_dir / class_name
            label = self.gloss_to_idx[class_name]
            for npy_file in sorted(class_dir.glob("*.npy")):
                self.samples.append(npy_file)
                self.labels.append(label)

        print(f"  [{split}] {len(self.samples)} samples, {len(self.glosses)} classes"
              f"{' (augmented)' if augment else ''}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        keypoints = np.load(self.samples[idx]).astype(np.float32)

        # Ensure correct shape
        if keypoints.shape[0] != config.SEQ_LEN:
            if keypoints.shape[0] < config.SEQ_LEN:
                pad = np.zeros((config.SEQ_LEN - keypoints.shape[0], config.FEATURE_DIM), dtype=np.float32)
                keypoints = np.concatenate([keypoints, pad], axis=0)
            else:
                indices = np.linspace(0, keypoints.shape[0] - 1, config.SEQ_LEN, dtype=int)
                keypoints = keypoints[indices]

        # 1. Normalize keypoints (shoulder-relative + hand min-max)
        keypoints = normalize_sequence(keypoints)

        # 2. Data augmentation (training only)
        if self.augment:
            keypoints = augment_sequence(keypoints, p=0.5)

        # 3. Add velocity features (frame-to-frame deltas)
        if config.USE_VELOCITY:
            keypoints = add_velocity_features(keypoints)

        x = torch.from_numpy(keypoints)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


# ── Training Loop ──────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"\nDevice: {device}")
    print(f"Keypoints dir: {config.KEYPOINTS_DIR}")
    print()

    # Load datasets (training with augmentation, validation without)
    train_ds = KeypointDataset(config.KEYPOINTS_DIR, split="train", augment=True)
    glosses = train_ds.glosses

    # Try to load validation set, fall back to train subset
    try:
        val_ds = KeypointDataset(config.KEYPOINTS_DIR, split="val", glosses=glosses)
    except FileNotFoundError:
        print("  [INFO] No val split found, using 20% of training data")
        n_val = max(1, len(train_ds) // 5)
        train_ds, val_ds = torch.utils.data.random_split(
            train_ds, [len(train_ds) - n_val, n_val]
        )

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    # Model (use FEATURE_DIM_FINAL to include velocity features)
    num_classes = len(glosses)
    model = SignLanguageLSTM(
        input_dim=config.FEATURE_DIM_FINAL,
        num_classes=num_classes,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel: {total_params:,} parameters, {num_classes} classes")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE
    )

    best_val_acc = 0.0
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 50)

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)

        train_loss /= train_total
        train_acc = train_correct / train_total

        # ── Validate ──
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                val_loss += loss.item() * x.size(0)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)

        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}/{args.epochs} | "
                f"Train: loss={train_loss:.4f} acc={train_acc:.3f} | "
                f"Val: loss={val_loss:.4f} acc={val_acc:.3f} | "
                f"LR={lr:.1e}"
            )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0

            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "glosses": glosses,
                "val_acc": val_acc,
                "config": {
                    "input_dim": config.FEATURE_DIM_FINAL,
                    "hidden_size": config.HIDDEN_SIZE,
                    "num_layers": config.NUM_LAYERS,
                    "num_classes": num_classes,
                    "seq_len": config.SEQ_LEN,
                    "use_velocity": config.USE_VELOCITY,
                },
            }
            torch.save(checkpoint, config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"\n  Early stopping at epoch {epoch} (patience={config.PATIENCE})")
                break

    print("-" * 50)
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {config.MODEL_PATH}")
    print(f"Classes: {glosses}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISL LSTM classifier")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    train(args)
