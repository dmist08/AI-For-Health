"""
train_model.py — LOPO Cross-Validation training for sleep apnea detection.

Usage:
    python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.pkl"
"""
import os
import argparse
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from models.cnn_model import SimpleCNN
from src.config import config
from src.utils import setup_logger, set_seed, get_device

logger = setup_logger("TrainModel")
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True, parents=True)


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BreathingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    """Stops training when validation loss stops improving."""
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.counter = 0
        self.best_loss = float('inf')
        self.stop = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─── Training Loop ────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X, y)
            total_loss += criterion(logits, y).item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(labels), np.array(preds)

def train_fold(fold: int, train_p: list, test_p: str,
               dataset: dict, device: torch.device):
    logger.info(f"\n{'='*50}\nFold {fold}/5 — Test: {test_p}\n{'='*50}")

    # ── Build arrays ──────────────────────────────────────────────────────────
    X_train = np.concatenate([dataset[p]["X"] for p in train_p])
    y_train = np.concatenate([dataset[p]["y"] for p in train_p])
    X_test  = dataset[test_p]["X"]
    y_test  = dataset[test_p]["y"]

    # ── Normalize per channel (train stats only — no leakage) ────────────────
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    # ── DataLoaders ───────────────────────────────────────────────────────────
    train_loader = DataLoader(BreathingDataset(X_train, y_train),
                              batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.DATALOADER_WORKERS)
    test_loader  = DataLoader(BreathingDataset(X_test, y_test),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.DATALOADER_WORKERS)

    # ── Model & Optimizer ─────────────────────────────────────────────────────
    model = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=0.01)

    # Class weights (inverse frequency) — handles imbalance in loss function
    counts = Counter(y_train.tolist())
    total = len(y_train)
    weights = torch.tensor([total / counts.get(i, 1) for i in range(2)],
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    early_stopping = EarlyStopping(patience=15)

    best_val_loss = float('inf')
    best_preds, best_labels = None, None

    # ── Epoch Loop ────────────────────────────────────────────────────────────
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss, val_y, val_preds = evaluate(model, test_loader, criterion, device)

        # Log every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            _, _, f1, _ = precision_recall_fscore_support(
                val_y, val_preds, average='binary', zero_division=0)
            logger.info(f"Epoch {epoch+1:3d} | "
                        f"Train Loss: {train_loss:.4f} | "
                        f"Val Loss: {val_loss:.4f} | "
                        f"Binary F1: {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds  = val_preds.copy()
            best_labels = val_y.copy()

        early_stopping.step(val_loss)
        if early_stopping.stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    return best_labels, best_preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dataset_path: str) -> None:
    set_seed(config.SEED)
    device = get_device()
    logger.info(f"Device: {device}")

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    participants = sorted(dataset.keys())
    logger.info(f"Participants: {participants}")

    all_y_true, all_y_pred = [], []
    fold_results = []

    for i, test_p in enumerate(participants):
        train_p = [p for p in participants if p != test_p]
        y_true, y_pred = train_fold(i + 1, train_p, test_p, dataset, device)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)

        fold_results.append({
            "Fold": i + 1, "Test_Participant": test_p,
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4)
        })

        # Per-fold confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Normal", "Event"],
                    yticklabels=["Normal", "Event"])
        ax.set_title(f"Fold {i+1} — Test: {test_p}")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / f"cm_fold_{i+1}_{test_p}.png")
        plt.close()

    # ── Aggregate Results ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50 + "\nLOPO CV COMPLETE\n" + "=" * 50)
    df = pd.DataFrame(fold_results)
    logger.info("\n" + df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "lopo_metrics.csv", index=False)

    o_acc = accuracy_score(all_y_true, all_y_pred)
    o_prec, o_rec, o_f1, _ = precision_recall_fscore_support(
        all_y_true, all_y_pred, average='binary', zero_division=0)
    logger.info(f"\nAggregate | Acc: {o_acc:.4f} | "
                f"Prec: {o_prec:.4f} | Rec: {o_rec:.4f} | F1: {o_f1:.4f}")

    cm_all = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Reds', ax=ax,
                xticklabels=["Normal", "Event"],
                yticklabels=["Normal", "Event"])
    ax.set_title("Aggregate Confusion Matrix (All Folds)")
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cm_aggregate.png")
    plt.close()
    logger.info(f"Results saved to {RESULTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default="Dataset/breathing_dataset.pkl")
    args = parser.parse_args()
    main(args.dataset_path)
