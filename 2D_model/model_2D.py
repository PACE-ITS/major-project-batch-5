import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x):
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        context = torch.sum(weights * x, dim=1)
        return context, weights

class CNN_BiLSTM_Attention(nn.Module):
    def __init__(self, num_classes=4):
        super(CNN_BiLSTM_Attention, self).__init__()
        
        # Load Pre-trained EfficientNet-B0
        # Weights are converted to 3-channel as standard backbones expect RGB
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Remove the last classifier head
        self.feature_extractor = self.backbone.features
        
        # Global Pooling to 1280 features
        self.pool = nn.AdaptiveAvgPool2d((4, 4)) # (B, 1280, 4, 4)
        
        # BiLSTM processing
        self.lstm = nn.LSTM(input_size=1280, hidden_size=256, num_layers=2, 
                            batch_first=True, bidirectional=True, dropout=0.3)
        
        # Attention
        self.attention = AttentionBlock(hidden_dim=512)
        
        # Final classification head
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Convert grayscale (1 channel) to RGB (3 channels) for EfficientNet
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
            
        x = self.feature_extractor(x) # (B, 1280, 4, 4)
        x = self.pool(x)
        
        # Reshape for LSTM: (B, 1280, 4, 4) -> (B, 16, 1280)
        B, C, H, W = x.size()
        x = x.view(B, C, H*W).permute(0, 2, 1) # (B, 16, 1280)
        
        # BiLSTM
        lstm_out, _ = self.lstm(x) # (B, 16, 512)
        
        # Attention
        context, _ = self.attention(lstm_out)
        
        # Output
        out = self.fc(context)
        return out
