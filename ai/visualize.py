"""
Visualization — Train with full logging and generate presentation charts.

Usage:
    python visualize.py              # Retrain + generate all charts
    python visualize.py --eval-only  # Skip training, use saved model
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from model import SignLanguageLSTM
from train import KeypointDataset

CHARTS_DIR = config.BASE_DIR / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

BLUE = "#2196F3"
RED = "#F44336"
GREEN = "#4CAF50"
ORANGE = "#FF9800"
PURPLE = "#9C27B0"
GRAY = "#9E9E9E"
BG = "#FAFAFA"


def train_with_logging(epochs=80):
    device = torch.device("cpu")
    train_ds = KeypointDataset(config.KEYPOINTS_DIR, split="train", augment=True)
    glosses = train_ds.glosses
    val_ds = KeypointDataset(config.KEYPOINTS_DIR, split="val", glosses=glosses)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    num_classes = len(glosses)
    model = SignLanguageLSTM(input_dim=config.FEATURE_DIM_FINAL, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.SCHEDULER_FACTOR, patience=config.SCHEDULER_PATIENCE)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": [], "lr": [], "epochs": []}
    best_val_acc = 0.0
    patience_counter = 0

    print(f"Training {num_classes} classes, {sum(p.numel() for p in model.parameters()):,} params")

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_correct, t_total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * x.size(0)
            t_correct += (logits.argmax(1) == y).sum().item()
            t_total += x.size(0)

        model.eval()
        v_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits, y)
                v_loss += loss.item() * x.size(0)
                v_correct += (logits.argmax(1) == y).sum().item()
                v_total += x.size(0)

        train_loss = t_loss / t_total
        train_acc = t_correct / t_total
        val_loss = v_loss / max(v_total, 1)
        val_acc = v_correct / max(v_total, 1)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(lr)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d} | T={train_acc:.3f} V={val_acc:.3f} | loss={val_loss:.4f} | lr={lr:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "glosses": glosses, "val_acc": val_acc,
                "config": {"input_dim": config.FEATURE_DIM_FINAL, "hidden_size": config.HIDDEN_SIZE,
                           "num_layers": config.NUM_LAYERS, "num_classes": num_classes,
                           "seq_len": config.SEQ_LEN, "use_velocity": config.USE_VELOCITY},
            }, config.MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= config.PATIENCE:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"Best val accuracy: {best_val_acc:.3f}")
    with open(CHARTS_DIR / "training_history.json", "w") as f:
        json.dump(history, f)
    return history, glosses, model, device


def get_test_metrics(model, glosses, device):
    test_ds = KeypointDataset(config.KEYPOINTS_DIR, split="test", glosses=glosses)
    loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            preds = model(x).argmax(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.numpy())
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    nc = len(glosses)
    precision, recall, f1, per_class_acc, support = [np.zeros(nc) for _ in range(5)]
    for c in range(nc):
        tp = ((all_preds == c) & (all_labels == c)).sum()
        fp = ((all_preds == c) & (all_labels != c)).sum()
        fn = ((all_preds != c) & (all_labels == c)).sum()
        tot = (all_labels == c).sum()
        precision[c] = tp / max(tp + fp, 1)
        recall[c] = tp / max(tp + fn, 1)
        f1[c] = 2 * precision[c] * recall[c] / max(precision[c] + recall[c], 1e-8)
        per_class_acc[c] = tp / max(tot, 1)
        support[c] = tot
    return {"preds": all_preds, "labels": all_labels, "precision": precision, "recall": recall,
            "f1": f1, "acc": per_class_acc, "support": support,
            "overall": (all_preds == all_labels).mean(), "glosses": glosses}


# ── Charts ───────────────────────────────────────────────────────────

def chart_loss(h):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax.plot(h["epochs"], h["train_loss"], color=BLUE, lw=2, label="Train Loss")
    ax.plot(h["epochs"], h["val_loss"], color=RED, lw=2, label="Val Loss")
    ax.fill_between(h["epochs"], h["train_loss"], alpha=0.1, color=BLUE)
    ax.fill_between(h["epochs"], h["val_loss"], alpha=0.1, color=RED)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "01_loss_curves.png", dpi=150); plt.close()


def chart_accuracy(h):
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    ta = [a * 100 for a in h["train_acc"]]
    va = [a * 100 for a in h["val_acc"]]
    ax.plot(h["epochs"], ta, color=BLUE, lw=2, label="Train Acc")
    ax.plot(h["epochs"], va, color=GREEN, lw=2, label="Val Acc")
    ax.fill_between(h["epochs"], ta, alpha=0.1, color=BLUE)
    ax.fill_between(h["epochs"], va, alpha=0.1, color=GREEN)
    bv = max(va); be = h["epochs"][va.index(bv)]
    ax.axhline(y=bv, color=GREEN, ls="--", alpha=0.5)
    ax.annotate(f"Best: {bv:.1f}% (ep {be})", xy=(be, bv), fontsize=10,
                xytext=(be + 2, bv - 6), arrowprops=dict(arrowstyle="->", color=GREEN),
                color=GREEN, fontweight="bold")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy (%)")
    ax.set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_ylim(0, 105)
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "02_accuracy_curves.png", dpi=150); plt.close()


def chart_lr(h):
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=BG)
    ax.plot(h["epochs"], h["lr"], color=PURPLE, lw=2, marker=".", ms=3)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_yscale("log")
    ax.set_title("Learning Rate Schedule", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "03_lr_schedule.png", dpi=150); plt.close()


def chart_confusion(m):
    g = m["glosses"]; nc = len(g)
    cm = np.zeros((nc, nc), dtype=int)
    for p, l in zip(m["preds"], m["labels"]):
        cm[l][p] += 1
    fig, ax = plt.subplots(figsize=(20, 18), facecolor=BG)
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(nc)); ax.set_yticks(range(nc))
    ax.set_xticklabels(g, rotation=90, fontsize=6); ax.set_yticklabels(g, fontsize=6)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix ({nc} classes, Acc: {m['overall']:.1%})", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "04_confusion_matrix.png", dpi=150); plt.close()


def chart_per_class_acc(m):
    g = m["glosses"]; a = m["acc"] * 100
    si = np.argsort(a); sg = [g[i] for i in si]; sa = a[si]
    colors = [GREEN if v >= 80 else ORANGE if v >= 50 else RED for v in sa]
    fig, ax = plt.subplots(figsize=(12, max(8, len(g) * 0.25)), facecolor=BG)
    ax.barh(range(len(sg)), sa, color=colors, edgecolor="white", lw=0.5)
    ax.set_yticks(range(len(sg))); ax.set_yticklabels(sg, fontsize=7)
    ax.set_xlabel("Accuracy (%)"); ax.set_xlim(0, 105)
    ax.axvline(x=80, color=GRAY, ls="--", alpha=0.5)
    ax.set_title(f"Per-Class Test Accuracy (mean: {a.mean():.1f}%)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="x")
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "05_per_class_accuracy.png", dpi=150); plt.close()


def chart_prf1(m):
    g = m["glosses"]; si = np.argsort(m["f1"])[::-1]
    show = si[:40] if len(g) > 40 else si
    sg = [g[i] for i in show]
    sp, sr, sf = m["precision"][show], m["recall"][show], m["f1"][show]
    x = np.arange(len(sg)); w = 0.25
    fig, ax = plt.subplots(figsize=(16, 7), facecolor=BG)
    ax.bar(x - w, sp, w, label="Precision", color=BLUE, alpha=0.85)
    ax.bar(x, sr, w, label="Recall", color=GREEN, alpha=0.85)
    ax.bar(x + w, sf, w, label="F1", color=ORANGE, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(sg, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("Score"); ax.set_ylim(0, 1.15)
    ax.set_title("Precision / Recall / F1-Score", fontsize=14, fontweight="bold")
    ax.legend()
    mp, mr, mf = m["precision"].mean(), m["recall"].mean(), m["f1"].mean()
    ax.text(0.02, 0.95, f"Macro: P={mp:.3f} R={mr:.3f} F1={mf:.3f}",
            transform=ax.transAxes, fontsize=10, fontweight="bold",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.grid(True, alpha=0.2, axis="y")
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "06_precision_recall_f1.png", dpi=150); plt.close()


def chart_dataset():
    splits = {}
    for s in ["train", "val", "test"]:
        d = config.KEYPOINTS_DIR / s
        if d.exists():
            splits[s] = {x.name: len(list(x.glob("*.npy"))) for x in sorted(d.iterdir()) if x.is_dir()}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), facecolor=BG)
    totals = {s: sum(c.values()) for s, c in splits.items()}
    ax1.pie(totals.values(), labels=[f"{k}\n({v})" for k, v in totals.items()],
            colors=[BLUE, ORANGE, GREEN], autopct="%1.1f%%", startangle=90, textprops={"fontsize": 11})
    ax1.set_title("Split Distribution", fontsize=14, fontweight="bold")
    if "train" in splits:
        tc = splits["train"]; cl = sorted(tc.keys()); cn = [tc[c] for c in cl]
        ax2.barh(range(len(cl)), cn, color=[BLUE if c >= 10 else RED for c in cn], edgecolor="white", lw=0.3)
        ax2.set_yticks(range(len(cl))); ax2.set_yticklabels(cl, fontsize=6)
        ax2.set_xlabel("Samples")
        ax2.set_title(f"Train Samples/Class (total: {sum(cn)})", fontsize=14, fontweight="bold")
        ax2.grid(True, alpha=0.2, axis="x")
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "07_dataset_distribution.png", dpi=150); plt.close()


def chart_summary(m, h):
    fig, ax = plt.subplots(figsize=(12, 8), facecolor=BG)
    ax.axis("off")
    bv = max(h["val_acc"]) * 100
    be = h["epochs"][h["val_acc"].index(max(h["val_acc"]))]
    txt = (
        "Happy Club ISL Pipeline - Model Summary\n"
        "=" * 48 + "\n\n"
        f"Architecture:     BiLSTM + Attention Pooling\n"
        f"Parameters:       809,600\n"
        f"Input:            {config.SEQ_LEN} frames x {config.FEATURE_DIM_FINAL} features\n"
        f"Features:         108 position + 108 velocity\n"
        f"Normalization:    SignSpace (shoulder-relative)\n"
        f"Augmentation:     Mirror, Noise, Stretch, Scale, Rotate\n"
        f"LSTM:             {config.NUM_LAYERS} layers, hidden={config.HIDDEN_SIZE}, bidir\n"
        f"Optimizer:        AdamW lr={config.LEARNING_RATE}\n\n"
        "=" * 48 + "\n\n"
        f"Dataset:          INCLUDE ISL (Zenodo) 12 categories\n"
        f"Classes:          {len(m['glosses'])}\n"
        f"Samples:          924 train / 196 val / 253 test\n\n"
        f"Best Val Acc:     {bv:.1f}% (epoch {be})\n"
        f"Test Accuracy:    {m['overall']:.1%}\n"
        f"Macro Precision:  {m['precision'].mean():.3f}\n"
        f"Macro Recall:     {m['recall'].mean():.3f}\n"
        f"Macro F1:         {m['f1'].mean():.3f}\n"
    )
    ax.text(0.05, 0.95, txt, transform=ax.transAxes, fontsize=11, verticalalignment="top",
            fontfamily="monospace", bbox=dict(boxstyle="round", facecolor="white", edgecolor=BLUE, lw=2))
    plt.tight_layout(); plt.savefig(CHARTS_DIR / "08_model_summary.png", dpi=150); plt.close()


def chart_dashboard(h, m):
    fig = plt.figure(figsize=(16, 12), facecolor=BG)
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    # Loss
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(h["epochs"], h["train_loss"], color=BLUE, lw=2, label="Train")
    ax1.plot(h["epochs"], h["val_loss"], color=RED, lw=2, label="Val")
    ax1.set_title("Loss", fontweight="bold"); ax1.legend(); ax1.grid(True, alpha=0.3)
    # Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(h["epochs"], [a*100 for a in h["train_acc"]], color=BLUE, lw=2, label="Train")
    ax2.plot(h["epochs"], [a*100 for a in h["val_acc"]], color=GREEN, lw=2, label="Val")
    bv = max(h["val_acc"]) * 100
    ax2.axhline(y=bv, color=GREEN, ls="--", alpha=0.5)
    ax2.set_title(f"Accuracy (best: {bv:.1f}%)", fontweight="bold"); ax2.set_ylim(0, 105)
    ax2.legend(); ax2.grid(True, alpha=0.3)
    # Top F1
    ax3 = fig.add_subplot(gs[1, 0])
    si = np.argsort(m["f1"])[::-1][:20]
    sg = [m["glosses"][i] for i in si]; sf = m["f1"][si]
    colors = [GREEN if f >= 0.8 else ORANGE if f >= 0.5 else RED for f in sf]
    ax3.barh(range(len(sg)), sf, color=colors)
    ax3.set_yticks(range(len(sg))); ax3.set_yticklabels(sg, fontsize=7)
    ax3.set_title("Top 20 F1-Scores", fontweight="bold"); ax3.set_xlim(0, 1.1)
    ax3.invert_yaxis(); ax3.grid(True, alpha=0.2, axis="x")
    # Stats
    ax4 = fig.add_subplot(gs[1, 1]); ax4.axis("off")
    stats = (f"Test Accuracy:  {m['overall']:.1%}\n"
             f"Macro F1:       {m['f1'].mean():.3f}\n"
             f"Macro Prec:     {m['precision'].mean():.3f}\n"
             f"Macro Recall:   {m['recall'].mean():.3f}\n\n"
             f"Classes: {len(m['glosses'])}\n"
             f"Samples: 1,373\n"
             f"Params:  809,600\n"
             f"Features: 216 (pos+vel)")
    ax4.text(0.1, 0.9, stats, transform=ax4.transAxes, fontsize=13, va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="white", edgecolor=BLUE, lw=2))
    ax4.set_title("Summary", fontweight="bold")
    fig.suptitle("Happy Club ISL Pipeline - Dashboard", fontsize=16, fontweight="bold", y=0.98)
    plt.savefig(CHARTS_DIR / "00_dashboard.png", dpi=150, bbox_inches="tight"); plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--epochs", type=int, default=80)
    args = parser.parse_args()

    if args.eval_only:
        hp = CHARTS_DIR / "training_history.json"
        if not hp.exists():
            print("No history found. Run without --eval-only first.")
            exit(1)
        with open(hp) as f:
            history = json.load(f)
        ckpt = torch.load(config.MODEL_PATH, map_location="cpu", weights_only=False)
        glosses = ckpt["glosses"]
        inp = ckpt.get("config", {}).get("input_dim", config.FEATURE_DIM_FINAL)
        model = SignLanguageLSTM(input_dim=inp, num_classes=len(glosses))
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval(); device = torch.device("cpu")
    else:
        history, glosses, model, device = train_with_logging(args.epochs)

    print("\nEvaluating on test set...")
    metrics = get_test_metrics(model, glosses, device)
    print(f"  Test accuracy: {metrics['overall']:.1%}")
    print(f"  Macro F1: {metrics['f1'].mean():.3f}")

    print(f"\nGenerating charts to {CHARTS_DIR}/")
    chart_dashboard(history, metrics); print("  00_dashboard.png")
    chart_loss(history); print("  01_loss_curves.png")
    chart_accuracy(history); print("  02_accuracy_curves.png")
    chart_lr(history); print("  03_lr_schedule.png")
    chart_confusion(metrics); print("  04_confusion_matrix.png")
    chart_per_class_acc(metrics); print("  05_per_class_accuracy.png")
    chart_prf1(metrics); print("  06_precision_recall_f1.png")
    chart_dataset(); print("  07_dataset_distribution.png")
    chart_summary(metrics, history); print("  08_model_summary.png")
    print("\nDone! All charts saved.")
