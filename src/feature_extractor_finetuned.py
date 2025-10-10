import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import sys
import os
sys.path.append('../baseline')
from my_xception import Xception
from my_efficientnet import EfficientNetB3

class FinetunedXceptionFeatureExtractor(nn.Module):
    """Load fine-tuned Xception and extract features before final FC layer"""
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load fine-tuned model
        self.model = Xception(num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        
        # Extract features BEFORE classifier (timm uses 'fc' or 'classifier')
        backbone = self.model.backbone
        if hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Linear):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, 'head') and isinstance(backbone.head, nn.Linear):
            backbone.head = nn.Identity()
        
        print(f"Loaded fine-tuned Xception from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  - Num features: {self.model.num_features}")
        
    def forward(self, x):
        return self.model(x)

class FinetunedEfficientNetB3FeatureExtractor(nn.Module):
    """Load fine-tuned EfficientNet-B3 and extract features before final FC layer"""
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load fine-tuned model
        self.model = EfficientNetB3(num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        self.model.load_state_dict(checkpoint['model'])
        
        # Extract features BEFORE classifier (timm uses 'classifier')
        backbone = self.model.backbone
        if hasattr(backbone, 'classifier') and isinstance(backbone.classifier, nn.Linear):
            backbone.classifier = nn.Identity()
        elif hasattr(backbone, 'fc') and isinstance(backbone.fc, nn.Linear):
            backbone.fc = nn.Identity()
        elif hasattr(backbone, 'head') and isinstance(backbone.head, nn.Linear):
            backbone.head = nn.Identity()
        
        print(f"Loaded fine-tuned EfficientNet-B3 from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  - Num features: {self.model.num_features}")
        
    def forward(self, x):
        return self.model(x)

def extract_features(model, dataloader, device):
    """Extract features from a model"""
    model.eval()
    features_list = []
    labels_list = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.numpy())
    
    features = np.vstack(features_list)
    labels = np.concatenate(labels_list)
    return features, labels

def extract_and_stack_features_finetuned(dataloader, device='cuda', 
                                          xception_ckpt='../baseline/xception_Dataset_best.pth',
                                          efficientnet_ckpt='../baseline/efficientnet_b3_Dataset_best.pth'):
    """
    Extract and stack features from FINE-TUNED models (not frozen pretrained)
    
    Args:
        dataloader: PyTorch DataLoader
        device: 'cuda' or 'cpu'
        xception_ckpt: Path to fine-tuned Xception .pth
        efficientnet_ckpt: Path to fine-tuned EfficientNet-B3 .pth
    
    Returns:
        stacked_features: (N, 3584) numpy array
        labels: (N,) numpy array
    """
    # Load fine-tuned models
    xception = FinetunedXceptionFeatureExtractor(xception_ckpt).to(device)
    efficientnet = FinetunedEfficientNetB3FeatureExtractor(efficientnet_ckpt).to(device)
    
    print("\nExtracting Xception features (fine-tuned)...")
    xception_features, labels = extract_features(xception, dataloader, device)
    
    print("Extracting EfficientNet-B3 features (fine-tuned)...")
    efficientnet_features, _ = extract_features(efficientnet, dataloader, device)
    
    # Stack features
    stacked_features = np.hstack([xception_features, efficientnet_features])
    print(f"Stacked features shape: {stacked_features.shape}")
    
    return stacked_features, labels

def save_features(features, labels, save_path):
    np.savez(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")

def load_features(load_path):
    data = np.load(load_path)
    return data['features'], data['labels']
