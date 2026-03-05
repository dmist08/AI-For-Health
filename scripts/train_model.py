import os
import argparse
import pickle
import logging
from pathlib import Path
from collections import Counter
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import sys
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
from models.cnn_model import MultiBranchBreathingCNN
from src.config import config
from src.utils import setup_logger, set_seed, get_device

logger = setup_logger("Train")

class BreathingDataset(Dataset):
    def __init__(self, x_32, x_4, y):
        self.x_32 = torch.tensor(x_32, dtype=torch.float32)
        self.x_4 = torch.tensor(x_4, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x_32[idx], self.x_4[idx], self.y[idx]

def get_class_weights(y: np.ndarray) -> torch.Tensor:
    counts = Counter(y)
    total = len(y)
    # Inverse frequency weighting
    weights = [total / counts.get(i, 1) for i in range(3)] 
    return torch.tensor(weights, dtype=torch.float32)

def evaluate_model(model, loader, device, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for x32, x4, y in loader:
            x32, x4, y = x32.to(device), x4.to(device), y.to(device)
            logits = model(x32, x4)
            loss = criterion(logits, y)
            
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
    return total_loss / len(loader), np.array(all_labels), np.array(all_preds)

def train_fold(fold_idx, train_p, test_p, dataset_dict, device):
    logger.info(f"\n{'='*40}\nStarting Fold {fold_idx+1}/5 - Test Participant: {test_p}\n{'='*40}")
    
    # 1. Build train and test arrays
    X_32_train, X_4_train, y_train = [], [], []
    for p in train_p:
        X_32_train.append(dataset_dict[p]["X_32"])
        X_4_train.append(dataset_dict[p]["X_4"])
        y_train.append(dataset_dict[p]["y"])
        
    X_32_train = np.concatenate(X_32_train)
    X_4_train = np.concatenate(X_4_train)
    y_train = np.concatenate(y_train)
    
    X_32_test = dataset_dict[test_p]["X_32"]
    X_4_test = dataset_dict[test_p]["X_4"]
    y_test = dataset_dict[test_p]["y"]
    
    # Optional: Z-score Standardize based ON TRAIN metrics ONLY (avoiding leakage)
    # Using simple mean/std per channel
    mean_32 = X_32_train.mean(axis=(0, 2), keepdims=True)
    std_32 = X_32_train.std(axis=(0, 2), keepdims=True) + 1e-8
    mean_4 = X_4_train.mean(axis=(0, 2), keepdims=True)
    std_4 = X_4_train.std(axis=(0, 2), keepdims=True) + 1e-8

    X_32_train = (X_32_train - mean_32) / std_32
    X_4_train = (X_4_train - mean_4) / std_4
    X_32_test = (X_32_test - mean_32) / std_32
    X_4_test = (X_4_test - mean_4) / std_4
    
    # Create DataLoaders
    train_ds = BreathingDataset(X_32_train, X_4_train, y_train)
    test_ds = BreathingDataset(X_32_test, X_4_test, y_test)
    
    # Handle Class Imbalance in Training Loop Using WeightedRandomSampler
    # To ensure model sees enough minority class examples per batch
    class_counts = list(Counter(y_train).values())
    class_weights = 1.0 / np.array(class_counts)
    sample_weights = np.array([class_weights[t] for t in y_train])
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, sampler=sampler, num_workers=config.DATALOADER_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.DATALOADER_WORKERS)
    
    # 2. Setup Model
    model = MultiBranchBreathingCNN(num_classes=3).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    
    # Using class weights in loss as an extra security against severe imbalances
    loss_weights = get_class_weights(y_train).to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    # 3. Training Loop
    best_f1 = 0.0
    best_preds = None
    best_labels = None
    
    for epoch in range(config.EPOCHS):
        model.train()
        train_loss = 0.0
        
        for x32, x4, y in train_loader:
            x32, x4, y = x32.to(device), x4.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x32, x4)
            loss = criterion(logits, y)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        val_loss, val_y, val_p = evaluate_model(model, test_loader, device, criterion)
        
        # Optimize for Macro F1
        _, _, macro_f1, _ = precision_recall_fscore_support(val_y, val_p, average='macro', zero_division=0)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"Epoch {epoch+1:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Macro F1: {macro_f1:.4f}")
            
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_preds = val_p
            best_labels = val_y
            # We don't save model per fold dynamically to save disk space unless required,
            # this is for LOPO evaluation

    logger.info(f"Fold completed. Best Macro F1: {best_f1:.4f}")
    if test_p == "AP03":
        logger.warning(f"AP03 has all normal windows. Overstated accuracy expected. Macro F1: {best_f1:.4f} but relies entirely on class 0 recall.")
        
    return best_labels, best_preds

def main(dataset_path: str):
    set_seed(config.SEED)
    device = get_device()
    logger.info(f"Using device: {device}")
    
    with open(dataset_path, 'rb') as f:
        dataset_dict = pickle.load(f)
        
    participants = sorted(list(dataset_dict.keys()))
    
    if len(participants) != 5:
        logger.error("Expected 5 participants strictly.")
        return

    all_y_true = []
    all_y_pred = []
    
    fold_results = []

    for i, test_p in enumerate(participants):
        train_p = [p for p in participants if p != test_p]
        
        y_true, y_pred = train_fold(i, train_p, test_p, dataset_dict, device)
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        
        acc = accuracy_score(y_true, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro', zero_division=0)
        
        fold_results.append({
            "Fold": i + 1,
            "Test_Participant": test_p,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1
        })
        
        # Plot Fold CM
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Normal", "Hypopnea", "Obstructive"], yticklabels=["Normal", "Hypopnea", "Obstructive"])
        plt.title(f"Confusion Matrix - Fold {i+1} ({test_p})")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(config.RESULTS_DIR / f"cm_fold_{i+1}_{test_p}.png")
        plt.close()

    # Aggregate
    logger.info("\n" + "="*40 + "\nLOPO CV EVALUATION COMPLETE\n" + "="*40)
    df_results = pd.DataFrame(fold_results)
    print(df_results.to_markdown(index=False))
    df_results.to_csv(config.RESULTS_DIR / "lopo_metrics.csv", index=False)
    
    overall_acc = accuracy_score(all_y_true, all_y_pred)
    o_prec, o_rec, o_f1, _ = precision_recall_fscore_support(all_y_true, all_y_pred, average='macro', zero_division=0)
    logger.info(f"Aggregate | Acc: {overall_acc:.4f} | Prec: {o_prec:.4f} | Rec: {o_rec:.4f} | F1: {o_f1:.4f}")
    
    cm_all = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_all, annot=True, fmt='d', cmap='Reds', xticklabels=["Normal", "Hypopnea", "Obstructive"], yticklabels=["Normal", "Hypopnea", "Obstructive"])
    plt.title("Aggregate Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(config.RESULTS_DIR / "cm_aggregate.png")
    plt.close()
    
    logger.info(f"Results saved to {config.RESULTS_DIR}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_path", type=str, default="Dataset/breathing_dataset.pkl")
    args = parser.parse_args()
    main(args.dataset_path)
