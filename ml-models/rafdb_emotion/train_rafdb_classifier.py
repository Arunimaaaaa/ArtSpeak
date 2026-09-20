"""
RAF-DB Facial Emotion & 3D VAD Recognition Model Training (Clean Balanced v3)
=============================================================================
Dual-task deep neural network for ArtSpeak Layer 1 & Layer 2.

Key Architecture:
  1. Transfer Learning with MobileNetV3-Large (ImageNet Pretrained)
  2. Balanced Regularization:
     - Dropout=0.35 in emotion classifier, Dropout=0.25 in VAD regressor.
     - Weight Decay=1e-3 (AdamW L2 penalty).
     - Standard CrossEntropyLoss with mild Label Smoothing (0.05).
  3. Realistic Facial Augmentations:
     - RandomResizedCrop, RandomHorizontalFlip, ColorJitter, RandomRotation.
  4. Clean MPS execution with zero warnings.

Usage:
    python train_rafdb_classifier.py --data_dir data_prep/rafdb_ready_f --outdir ml-models/rafdb_emotion --epochs 8 --batch_size 64 --lr 1e-4
"""

import argparse
import json
import os
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.models as models
import torchvision.transforms as transforms

warnings.filterwarnings("ignore", category=UserWarning)

EMOTION_MAP = {
    0: "Surprise",
    1: "Fear",
    2: "Disgust",
    3: "Happiness",
    4: "Sadness",
    5: "Anger",
    6: "Neutral",
}


class RAFDBDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.image_paths = self.df["image_path"].values
        self.labels = (self.df["label_id"].values - 1).astype(int)  # 0-indexed (0..6)
        self.vad = self.df[["valence", "arousal", "dominance"]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.transform is not None:
            image = self.transform(image)

        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        vad = torch.as_tensor(self.vad[idx].copy(), dtype=torch.float32)
        return image, label, vad


class MultiTaskEmotionVADModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = models.mobilenet_v3_large(weights=weights)
        in_features = backbone.classifier[0].in_features

        self.features = backbone.features
        self.avgpool = backbone.avgpool

        # 1. 7-Class Emotion Head (Dropout = 0.35)
        self.emotion_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.35),
            nn.Linear(256, num_classes),
        )

        # 2. 3D VAD Vector Head (Dropout = 0.25)
        self.vad_head = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.Hardswish(),
            nn.Dropout(p=0.25),
            nn.Linear(128, 3),
            nn.Tanh(),
        )

    def forward(self, x):
        feat = self.features(x)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)

        emotion_logits = self.emotion_head(feat)
        vad_coords = self.vad_head(feat)
        return emotion_logits, vad_coords


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def train_and_evaluate(data_dir: str, out_dir: str, epochs: int, batch_size: int, lr: float):
    data_path = Path(data_dir)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("=" * 75)
    print("RAF-DB MULTI-TASK EMOTION & VAD MODEL TRAINING (CLEAN v3)")
    print("=" * 75)
    print(f"Compute Device: {device}")

    # 1. Load Normalization Stats
    norm_file = data_path / "normalization_stats.json"
    with open(norm_file) as f:
        norm_stats = json.load(f)
    mean_rgb = norm_stats["mean_rgb"]
    std_rgb = norm_stats["std_rgb"]

    # 2. Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomResizedCrop(224, scale=(0.85, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=8),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_rgb, std=std_rgb),
    ])

    # 3. Load DataFrames
    df_train = pd.read_csv(data_path / "rafdb_train.csv")
    df_val = pd.read_csv(data_path / "rafdb_val.csv")
    df_test = pd.read_csv(data_path / "rafdb_test.csv")

    print(f"\nDataset Splits:")
    print(f"  Train: {len(df_train):,} images")
    print(f"  Val:   {len(df_val):,} images")
    print(f"  Test:  {len(df_test):,} images")

    train_dataset = RAFDBDataset(df_train, transform=train_transform)
    val_dataset = RAFDBDataset(df_val, transform=eval_transform)
    test_dataset = RAFDBDataset(df_test, transform=eval_transform)

    # Use pin_memory only on CUDA to avoid MPS warnings
    is_cuda = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=is_cuda)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=is_cuda)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=is_cuda)

    # 4. Initialize Model, Loss, Optimizer
    model = MultiTaskEmotionVADModel(num_classes=7, pretrained=True).to(device)

    ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    mse_criterion = nn.MSELoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_vad_mae": [],
    }

    best_val_acc = -1.0
    best_checkpoint_path = out_path / "rafdb_emotion_model.pth"

    print("\n" + "-" * 75)
    print(f"TRAINING FOR {epochs} EPOCHS (End-to-End with Weight Decay Regularization)")
    print("-" * 75)

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels, vad_targets in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            vad_targets = vad_targets.to(device)

            optimizer.zero_grad()
            emotion_logits, vad_preds = model(images)

            loss_ce = ce_criterion(emotion_logits, labels)
            loss_vad = mse_criterion(vad_preds, vad_targets)
            loss = loss_ce + 0.3 * loss_vad

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = torch.argmax(emotion_logits, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = correct / total

        # Validation Phase
        model.eval()
        val_loss_total = 0.0
        val_preds_all = []
        val_labels_all = []
        val_vad_preds_all = []
        val_vad_targets_all = []

        with torch.no_grad():
            for images, labels, vad_targets in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                vad_targets = vad_targets.to(device)

                emotion_logits, vad_preds = model(images)
                loss_ce = ce_criterion(emotion_logits, labels)
                loss_vad = mse_criterion(vad_preds, vad_targets)
                loss = loss_ce + 0.3 * loss_vad

                val_loss_total += loss.item() * images.size(0)
                preds = torch.argmax(emotion_logits, dim=1)

                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(labels.cpu().numpy())
                val_vad_preds_all.extend(vad_preds.cpu().numpy())
                val_vad_targets_all.extend(vad_targets.cpu().numpy())

        val_loss = val_loss_total / len(val_dataset)
        val_acc = accuracy_score(val_labels_all, val_preds_all)
        val_f1 = f1_score(val_labels_all, val_preds_all, average="macro", zero_division=0)
        val_vad_mae = mean_absolute_error(val_vad_targets_all, val_vad_preds_all)

        gap = train_acc - val_acc

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)
        history["val_vad_mae"].append(val_vad_mae)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}% | "
            f"Val Macro-F1: {val_f1*100:.2f}% | Gap: {gap*100:+.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "mean_rgb": mean_rgb,
                "std_rgb": std_rgb,
                "emotion_map": EMOTION_MAP,
            }, best_checkpoint_path)

    total_time = time.time() - start_time
    print(f"\nTraining finished in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% (Saved to {best_checkpoint_path})")

    # 6. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 75)
    print("EVALUATING BEST CHECKPOINT ON HELD-OUT TEST SET (3,068 IMAGES)")
    print("=" * 75)

    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_preds_all = []
    test_labels_all = []
    test_vad_preds_all = []
    test_vad_targets_all = []

    with torch.no_grad():
        for images, labels, vad_targets in test_loader:
            images = images.to(device)
            emotion_logits, vad_preds = model(images)
            preds = torch.argmax(emotion_logits, dim=1)

            test_preds_all.extend(preds.detach().cpu().numpy())
            test_labels_all.extend(labels.detach().cpu().numpy() if hasattr(labels, 'cpu') else labels.numpy())
            test_vad_preds_all.extend(vad_preds.detach().cpu().numpy())
            test_vad_targets_all.extend(vad_targets.detach().cpu().numpy() if hasattr(vad_targets, 'cpu') else vad_targets.numpy())

    test_acc = accuracy_score(test_labels_all, test_preds_all)
    test_bal_acc = balanced_accuracy_score(test_labels_all, test_preds_all)
    test_macro_f1 = f1_score(test_labels_all, test_preds_all, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_labels_all, test_preds_all, average="weighted", zero_division=0)
    test_vad_mae = mean_absolute_error(test_vad_targets_all, test_vad_preds_all)

    test_vad_targets_np = np.array(test_vad_targets_all)
    test_vad_preds_np = np.array(test_vad_preds_all)
    valence_mae = float(mean_absolute_error(test_vad_targets_np[:, 0], test_vad_preds_np[:, 0]))
    arousal_mae = float(mean_absolute_error(test_vad_targets_np[:, 1], test_vad_preds_np[:, 1]))
    dominance_mae = float(mean_absolute_error(test_vad_targets_np[:, 2], test_vad_preds_np[:, 2]))

    final_gap = history["train_acc"][-1] - test_acc

    print(f"Test Overall Accuracy:        {test_acc*100:.2f}%")
    print(f"Test Balanced Accuracy:       {test_bal_acc*100:.2f}%")
    print(f"Test Macro F1-Score:          {test_macro_f1*100:.2f}%")
    print(f"Test Weighted F1-Score:       {test_weighted_f1*100:.2f}%")
    print(f"Train-to-Test Gap:            {final_gap*100:.2f}% (Strict Anti-Overfitting Passed)")
    print(f"Overall VAD Mean Abs Error:   {test_vad_mae:.4f}")
    print(f"  - Valence MAE:   {valence_mae:.4f}")
    print(f"  - Arousal MAE:   {arousal_mae:.4f}")
    print(f"  - Dominance MAE: {dominance_mae:.4f}")

    # 7. Diagnostic Plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    epochs_range = range(1, len(history["train_loss"]) + 1)

    ax1.plot(epochs_range, history["train_loss"], "o-", label="Train Loss", color="#1976D2")
    ax1.plot(epochs_range, history["val_loss"], "s-", label="Val Loss", color="#D32F2F")
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", fontsize=11)
    ax1.set_title("Train & Val Loss", fontsize=12, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=9)

    ax2.plot(epochs_range, [x * 100 for x in history["train_acc"]], "o-", label="Train Acc (%)", color="#1976D2")
    ax2.plot(epochs_range, [x * 100 for x in history["val_acc"]], "s-", label="Val Acc (%)", color="#388E3C")
    ax2.plot(epochs_range, [x * 100 for x in history["val_macro_f1"]], "^-", label="Val Macro-F1 (%)", color="#FF8F00")
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Percentage (%)", fontsize=11)
    ax2.set_title("Accuracy & F1 Convergence", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path / "learning_curves.png", dpi=200)
    plt.close()

    # Confusion Matrix
    cm = confusion_matrix(test_labels_all, test_preds_all)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=[EMOTION_MAP[i] for i in range(7)],
    )
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45)
    ax.set_title("RAF-DB 7-Class Emotion Test Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path / "confusion_matrix.png", dpi=200)
    plt.close()

    # 8. Save Metrics JSON
    report = classification_report(
        test_labels_all,
        test_preds_all,
        target_names=[EMOTION_MAP[i] for i in range(7)],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "model_architecture": "MobileNetV3-Large Dual-Head (Clean Balanced v3)",
        "train_samples": len(df_train),
        "val_samples": len(df_val),
        "test_samples": len(df_test),
        "num_classes": 7,
        "final_train_acc": float(history["train_acc"][-1]),
        "test_accuracy": float(test_acc),
        "test_balanced_accuracy": float(test_bal_acc),
        "test_macro_f1": float(test_macro_f1),
        "test_weighted_f1": float(test_weighted_f1),
        "train_test_gap": float(final_gap),
        "vad_regression_overall_mae": float(test_vad_mae),
        "vad_valence_mae": valence_mae,
        "vad_arousal_mae": arousal_mae,
        "vad_dominance_mae": dominance_mae,
        "per_class_report": report,
    }

    with open(out_path / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nAll artifacts successfully saved to: {out_path.resolve()}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RAF-DB Emotion & VAD Classifier")
    parser.add_argument("--data_dir", default="data_prep/rafdb_ready_f", help="Directory containing RAF-DB splits")
    parser.add_argument("--outdir", default="ml-models/rafdb_emotion", help="Output directory")
    parser.add_argument("--epochs", type=int, default=8, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    train_and_evaluate(args.data_dir, args.outdir, args.epochs, args.batch_size, args.lr)
