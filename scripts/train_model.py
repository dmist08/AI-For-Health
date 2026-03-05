"""
train_model.py — LOPO Cross-Validation with multi-mode binary classification.

Modes:
    event     — Normal vs Event (Hypopnea OR Apnea)       [default]
    hypopnea  — Normal vs Hypopnea only (drops Apnea windows)
    apnea     — Normal vs Apnea only    (drops Hypopnea windows)

Usage:
    python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.pkl" -mode event
    python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.pkl" -mode hypopnea
    python scripts/train_model.py -dataset_path "Dataset/breathing_dataset.pkl" -mode apnea
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

# ─── Label Mappings ───────────────────────────────────────────────────────────

def get_binary_label(y_str: str, mode: str) -> int | None:
    """
    Returns 0 (Normal), 1 (Event), or None (window should be excluded).
    None means this window is irrelevant for this mode and should be dropped.
    """
    label = y_str.strip()
    if mode == "event":
        return 0 if label == "Normal" else 1
    elif mode == "hypopnea":
        if label == "Normal":       return 0
        if label == "Hypopnea":     return 1
        return None  # Drop Apnea windows
    elif mode == "apnea":
        if label == "Normal":       return 0
        if "Apnea" in label:        return 1
        return None  # Drop Hypopnea windows
    raise ValueError(f"Unknown mode: {mode}")


# ─── Dataset ──────────────────────────────────────────────────────────────────

class BreathingDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ─── Helper: Apply Mode Filter to a Participant's Data ───────────────────────

def apply_mode(p_data: dict, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Filter and relabel one participant's windows based on mode."""
    X_all   = p_data["X"]
    y_str   = p_data["y_str"]

    X_out, y_out = [], []
    for x, ys in zip(X_all, y_str):
        label = get_binary_label(ys, mode)
        if label is not None:
            X_out.append(x)
            y_out.append(label)

    if not X_out:
        return np.empty((0, *X_all.shape[1:]), dtype=np.float32), np.empty(0, dtype=np.int64)

    return np.array(X_out, dtype=np.float32), np.array(y_out, dtype=np.int64)


# ─── Early Stopping ───────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience: int = 15):
        self.patience = patience
        self.counter  = 0
        self.best     = float('inf')
        self.stop     = False

    def step(self, val_loss: float) -> None:
        if val_loss < self.best:
            self.best, self.counter = val_loss, 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ─── Evaluate ─────────────────────────────────────────────────────────────────

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, preds, labels = 0.0, [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            total_loss += criterion(logits, y).item()
            preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return total_loss / len(loader), np.array(labels), np.array(preds)


# ─── Train One Fold ───────────────────────────────────────────────────────────

def train_fold(fold: int, train_p: list, test_p: str,
               dataset: dict, mode: str, device: torch.device,
               results_dir: Path):

    logger.info(f"\n{'='*50}\nFold {fold}/5 — Test: {test_p}  [mode={mode}]\n{'='*50}")

    # Build train/test arrays with mode filtering
    X_parts, y_parts = [], []
    for p in train_p:
        Xp, yp = apply_mode(dataset[p], mode)
        if len(Xp): X_parts.append(Xp); y_parts.append(yp)

    if not X_parts:
        logger.error(f"No training data for fold {fold} with mode={mode}. Skipping.")
        return None, None

    X_train = np.concatenate(X_parts)
    y_train = np.concatenate(y_parts)
    X_test, y_test = apply_mode(dataset[test_p], mode)

    if len(X_test) == 0:
        logger.warning(f"No test data for {test_p} with mode={mode}. Skipping fold.")
        return None, None

    # Normalize per channel (train stats only — no leakage)
    mean = X_train.mean(axis=(0, 2), keepdims=True)
    std  = X_train.std(axis=(0, 2),  keepdims=True) + 1e-8
    X_train = (X_train - mean) / std
    X_test  = (X_test  - mean) / std

    counts = Counter(y_train.tolist())
    logger.info(f"Train: {len(X_train)} windows {dict(counts)} | Test: {len(X_test)} windows")

    train_loader = DataLoader(BreathingDataset(X_train, y_train),
                              batch_size=config.BATCH_SIZE, shuffle=True,
                              num_workers=config.DATALOADER_WORKERS)
    test_loader  = DataLoader(BreathingDataset(X_test, y_test),
                              batch_size=config.BATCH_SIZE, shuffle=False,
                              num_workers=config.DATALOADER_WORKERS)

    model     = SimpleCNN(num_classes=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE,
                                  weight_decay=0.01)

    # Class weights for imbalance
    total   = len(y_train)
    weights = torch.tensor([total / counts.get(i, 1) for i in range(2)],
                           dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    es = EarlyStopping(patience=15)
    best_val_loss = float('inf')
    best_preds, best_labels = None, None
    best_state = None

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

        val_loss, val_y, val_p = evaluate(model, test_loader, criterion, device)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            _, _, f1, _ = precision_recall_fscore_support(
                val_y, val_p, average='binary', zero_division=0)
            logger.info(f"Ep {epoch+1:3d} | TLoss {train_loss:.4f} | "
                        f"VLoss {val_loss:.4f} | F1 {f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_preds  = val_p.copy()
            best_labels = val_y.copy()
            best_state  = {k: v.clone() for k, v in model.state_dict().items()}

        es.step(val_loss)
        if es.stop:
            logger.info(f"Early stopping at epoch {epoch + 1}")
            break

    # Save best model weights for this fold
    model_path = results_dir / f"model_{mode}_fold{fold}_{test_p}.pt"
    torch.save(best_state, model_path)
    logger.info(f"Model saved: {model_path}")

    return best_labels, best_preds


# ─── Main ─────────────────────────────────────────────────────────────────────

def main(dataset_path: str, mode: str) -> None:
    set_seed(config.SEED)
    device = get_device()
    logger.info(f"Device: {device} | Mode: {mode}")

    with open(dataset_path, "rb") as f:
        dataset = pickle.load(f)

    participants = sorted(dataset.keys())
    logger.info(f"Participants: {participants}")

    # Results sub-directory per mode
    results_dir = PROJECT_ROOT / "results" / mode
    results_dir.mkdir(parents=True, exist_ok=True)

    all_y_true, all_y_pred = [], []
    fold_results = []

    for i, test_p in enumerate(participants):
        train_p = [p for p in participants if p != test_p]
        y_true, y_pred = train_fold(i + 1, train_p, test_p, dataset,
                                    mode, device, results_dir)

        if y_true is None:
            continue

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

        acc  = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average='binary', zero_division=0)

        fold_results.append({
            "Fold": i + 1, "Test": test_p, "Mode": mode,
            "Accuracy": round(acc, 4), "Precision": round(prec, 4),
            "Recall": round(rec, 4), "F1": round(f1, 4)
        })

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=["Normal", "Event"],
                    yticklabels=["Normal", "Event"])
        ax.set_title(f"[{mode}] Fold {i+1} — Test: {test_p}")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(results_dir / f"cm_fold_{i+1}_{test_p}.png")
        plt.close()

    # Aggregate
    logger.info(f"\n{'='*50}\nLOPO CV COMPLETE — mode={mode}\n{'='*50}")
    df = pd.DataFrame(fold_results)
    logger.info("\n" + df.to_string(index=False))
    df.to_csv(results_dir / "lopo_metrics.csv", index=False)

    if all_y_true:
        o_acc = accuracy_score(all_y_true, all_y_pred)
        o_prec, o_rec, o_f1, _ = precision_recall_fscore_support(
            all_y_true, all_y_pred, average='binary', zero_division=0)
        logger.info(f"Aggregate | Acc: {o_acc:.4f} | Prec: {o_prec:.4f} | "
                    f"Rec: {o_rec:.4f} | F1: {o_f1:.4f}")

        cm_all = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm_all, annot=True, fmt='d', cmap='Reds', ax=ax,
                    xticklabels=["Normal", "Event"],
                    yticklabels=["Normal", "Event"])
        ax.set_title(f"Aggregate CM — {mode}")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(results_dir / "cm_aggregate.png")
        plt.close()

    logger.info(f"All results saved to {results_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str,
                        default="Dataset/breathing_dataset.pkl")
    parser.add_argument("-mode", type=str, default="event",
                        choices=["event", "hypopnea", "apnea"],
                        help="Binary classification mode")
    args = parser.parse_args()
    main(args.dataset_path, args.mode)
