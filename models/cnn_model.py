"""
cnn_model.py — 1D CNN architecture for sleep apnea detection.

Input:  (Batch, 3, 960) — 3 channels (Flow, Thoracic, SpO2 upsampled), 30s at 32Hz
Output: (Batch, num_classes)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """Conv1D → BatchNorm → ReLU → MaxPool → Dropout."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int, pool: int,
                 dropout: float = 0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=kernel, padding='same'),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.MaxPool1d(pool),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SimpleCNN(nn.Module):
    """
    Single-branch 1D CNN for binary breathing event classification.

    All 3 signals (Nasal Airflow, Thoracic, SpO2) are resampled to 32Hz
    and stacked as separate channels → input shape: (Batch, 3, 960).

    Architecture:
        3 ConvBlocks with progressive downsampling
        → Global Average Pooling
        → Dropout
        → Linear classifier
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(in_ch=3,  out_ch=32,  kernel=7, pool=4),  # → (B, 32,  240)
            ConvBlock(in_ch=32, out_ch=64,  kernel=5, pool=4),  # → (B, 64,   60)
            ConvBlock(in_ch=64, out_ch=128, kernel=3, pool=4),  # → (B, 128,  15)
        )
        # Global Average Pooling collapses time dimension → (B, 128)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 128, 15)
        x = x.mean(dim=2)             # Global Average Pool → (B, 128)
        return self.classifier(x)     # (B, num_classes)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = SimpleCNN(num_classes=2)
    print(f"Parameters: {count_parameters(model):,}")
    dummy = torch.randn(8, 3, 960)
    out = model(dummy)
    print(f"Output shape: {out.shape}")  # Expect: (8, 2)
