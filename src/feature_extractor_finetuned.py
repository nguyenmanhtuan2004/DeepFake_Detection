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
        # Load fine-tuned model với num_classes=2 để load checkpoint
        wrapper = Xception(num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        wrapper.load_state_dict(checkpoint['model'])
        
        # Tạo model MỚI với num_classes=0 để trích xuất features
        # timm sẽ tự động loại bỏ classifier khi num_classes=0
        import timm
        model_name = wrapper.model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # ← Quan trọng: num_classes=0 để lấy features!
            global_pool="avg"
        )
        
        # Copy weights từ fine-tuned model (trừ classifier head)
        wrapper_state = wrapper.backbone.state_dict()
        model_state = self.backbone.state_dict()
        
        # Chỉ copy các layers có shape giống nhau (bỏ qua classifier)
        filtered_state = {k: v for k, v in wrapper_state.items() 
                         if k in model_state and v.shape == model_state[k].shape}
        self.backbone.load_state_dict(filtered_state, strict=False)
        
        num_features = self.backbone.num_features
        print(f"Loaded fine-tuned Xception from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  - Num features: {num_features}")
        
    def forward(self, x):
        return self.backbone(x)

class FinetunedEfficientNetB3FeatureExtractor(nn.Module):
    """Load fine-tuned EfficientNet-B3 and extract features before final FC layer"""
    def __init__(self, checkpoint_path):
        super().__init__()
        # Load fine-tuned model với num_classes=2 để load checkpoint
        wrapper = EfficientNetB3(num_classes=2, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        wrapper.load_state_dict(checkpoint['model'])
        
        # Tạo model MỚI với num_classes=0 để trích xuất features
        # timm sẽ tự động loại bỏ classifier khi num_classes=0
        import timm
        model_name = wrapper.model_name
        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0,  # ← Quan trọng: num_classes=0 để lấy features!
            global_pool="avg"
        )
        
        # Copy weights từ fine-tuned model (trừ classifier head)
        wrapper_state = wrapper.backbone.state_dict()
        model_state = self.backbone.state_dict()
        
        # Chỉ copy các layers có shape giống nhau (bỏ qua classifier)
        filtered_state = {k: v for k, v in wrapper_state.items() 
                         if k in model_state and v.shape == model_state[k].shape}
        self.backbone.load_state_dict(filtered_state, strict=False)
        
        num_features = self.backbone.num_features
        print(f"Loaded fine-tuned EfficientNet-B3 from {checkpoint_path}")
        print(f"  - Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  - Val Acc: {checkpoint.get('val_acc', 'N/A'):.4f}")
        print(f"  - Num features: {num_features}")
        
    def forward(self, x):
        return self.backbone(x)

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
    # Load fine-tuned models
    xception = FinetunedXceptionFeatureExtractor(xception_ckpt).to(device)
    efficientnet = FinetunedEfficientNetB3FeatureExtractor(efficientnet_ckpt).to(device)
    
    print("\nExtracting Xception features (fine-tuned)...")
    xception_features, labels = extract_features(xception, dataloader, device)
    print(f"  → Xception features shape: {xception_features.shape}")
    
    print("Extracting EfficientNet-B3 features (fine-tuned)...")
    efficientnet_features, _ = extract_features(efficientnet, dataloader, device)
    print(f"  → EfficientNet features shape: {efficientnet_features.shape}")
    
    # Stack features
    stacked_features = np.hstack([xception_features, efficientnet_features])
    print(f"  → Stacked features shape: {stacked_features.shape}")
    
    return stacked_features, labels

def save_features(features, labels, save_path):
    np.savez(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")

def load_features(load_path):
    data = np.load(load_path)
    return data['features'], data['labels']
