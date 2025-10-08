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

class EfficientNetB7FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Dùng tf_efficientnet_b7 vì có pretrained weights
        self.model = timm.create_model('tf_efficientnet_b7', pretrained=pretrained)
        self.model.classifier = nn.Identity()
        
    def forward(self, x):
        return self.model(x)

class MetaLearnerMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, num_classes=2, dropout=0.5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        return self.mlp(x)
