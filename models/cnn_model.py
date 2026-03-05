import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, pool_size: int):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pool(x)
        return self.dropout(x)

class MultiBranchBreathingCNN(nn.Module):
    """
    1D CNN combining 32Hz (Flow, Thoracic) and 4Hz (SpO2) inputs.
    Both branches utilize global average pooling before concatenation.
    """
    def __init__(self, num_classes: int = 3):
        super().__init__()
        
        # Branch A: High Frequency (Flow + Thoracic) - Input Shape: (Batch, 2, 960)
        self.branch_a = nn.Sequential(
            ConvBlock(in_channels=2, out_channels=32, kernel_size=7, pool_size=4),  # out: (Batch, 32, 240)
            ConvBlock(in_channels=32, out_channels=64, kernel_size=5, pool_size=4), # out: (Batch, 64, 60)
            ConvBlock(in_channels=64, out_channels=128, kernel_size=3, pool_size=4) # out: (Batch, 128, 15)
        )
        
        # Branch B: Low Frequency (SpO2) - Input Shape: (Batch, 1, 120)
        # Because sequence is shorter, we pool less aggressively
        self.branch_b = nn.Sequential(
            ConvBlock(in_channels=1, out_channels=16, kernel_size=5, pool_size=2),  # out: (Batch, 16, 60)
            ConvBlock(in_channels=16, out_channels=32, kernel_size=3, pool_size=2), # out: (Batch, 32, 30)
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, pool_size=2)  # out: (Batch, 64, 15)
        )
        
        # Global Average Pooling applied in `forward` manually to match varying time steps if any
        
        # Combined dense features: 128 (Branch A) + 64 (Branch B) = 192
        self.classifier = nn.Sequential(
            nn.Linear(192, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_32, x_4):
        # Forward pass through branches
        feat_a = self.branch_a(x_32)
        feat_b = self.branch_b(x_4)
        
        # Global Average Pooling (B, C, L) -> (B, C)
        feat_a_pooled = feat_a.mean(dim=2)
        feat_b_pooled = feat_b.mean(dim=2)
        
        # Concatenate features
        combined_features = torch.cat([feat_a_pooled, feat_b_pooled], dim=1)
        
        # Classification head
        logits = self.classifier(combined_features)
        return logits

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    # Smoke test the architecture dimensions
    model = MultiBranchBreathingCNN(num_classes=3)
    print(f"Model Parameters: {count_parameters(model):,}")
    
    dummy_x32 = torch.randn(8, 2, 960) # Batch=8, Channels(Flow, Thorac)=2, Len=960
    dummy_x4 = torch.randn(8, 1, 120)  # Batch=8, Channel(SpO2)=1, Len=120
    
    out = model(dummy_x32, dummy_x4)
    print(f"Output Shape: {out.shape}") # Should be (8, 3)
