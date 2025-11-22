import torch
import torch.nn as nn
from torchvision import models
import timm

class XceptionFeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('xception', pretrained=pretrained)
        self.model.fc = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class EfficientNetB3FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        self.model = timm.create_model('efficientnet_b3', pretrained=pretrained)
        self.model.classifier = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class MetaLearnerMLP(nn.Module):
    def __init__(self, input_dim, num_classes=2, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            # Layer 1: input_dim → 1024
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            
            # Layer 2: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 3: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Layer 4: 256 → 128
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            # Output: 128 → 2
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)
