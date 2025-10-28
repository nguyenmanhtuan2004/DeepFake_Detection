# resnet.py — PyTorch torchvision thuần túy
import torch
import torch.nn as nn
from torchvision import models

__all__ = ["ResNet50", "ResNet50_pretrained"]

class ResNet50(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        
        # Load ResNet50 từ torchvision
        if pretrained:
            self.backbone = models.resnet50(weights='DEFAULT')
        else:
            self.backbone = models.resnet50(weights=None)
        
        # Lấy số features từ fc hiện tại
        self.num_features = self.backbone.fc.in_features
        
        # Thay thế fc layer
        self.backbone.fc = nn.Linear(self.num_features, num_classes)
        
        if freeze_backbone:
            self.freeze_backbone()

    # ---------- freeze / unfreeze ----------
    def freeze_backbone(self):
        """
        Freeze toàn bộ backbone trừ layer3, layer4 và fc.
        Chiến lược fine-tune: train từ layer3 trở đi.
        """
        # Freeze toàn bộ trước
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze fc (head)
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
        
        # Unfreeze layer4 (deep features)
        for param in self.backbone.layer4.parameters():
            param.requires_grad = True
        
        # Unfreeze layer3 (mid-level features)
        for param in self.backbone.layer3.parameters():
            param.requires_grad = True

    def unfreeze(self):
        """Unfreeze toàn bộ model"""
        for param in self.backbone.parameters():
            param.requires_grad = True

    # ---------- forward ----------
    def forward(self, x):
        return self.backbone(x)



