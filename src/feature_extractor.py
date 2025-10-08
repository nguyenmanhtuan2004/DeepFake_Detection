import torch
import numpy as np
from tqdm import tqdm
from models import XceptionFeatureExtractor, EfficientNetB3FeatureExtractor, EfficientNetB7FeatureExtractor

def extract_features(model, dataloader, device):
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

def extract_and_stack_features(dataloader, device='cuda', use_b7=False):
    xception = XceptionFeatureExtractor(pretrained=True).to(device)
    
    if use_b7:
        print("Using EfficientNet-B7 (larger model, better features)")
        efficientnet = EfficientNetB7FeatureExtractor(pretrained=True).to(device)
        model_name = "B7"
    else:
        print("Using EfficientNet-B3")
        efficientnet = EfficientNetB3FeatureExtractor(pretrained=True).to(device)
        model_name = "B3"
    
    print("Extracting Xception features...")
    xception_features, labels = extract_features(xception, dataloader, device)
    
    print(f"Extracting EfficientNet-{model_name} features...")
    efficientnet_features, _ = extract_features(efficientnet, dataloader, device)
    
    stacked_features = np.hstack([xception_features, efficientnet_features])
    print(f"Stacked features shape: {stacked_features.shape}")
 
    
    return stacked_features, labels

def save_features(features, labels, save_path):
    np.savez(save_path, features=features, labels=labels)
    print(f"Features saved to {save_path}")

def load_features(load_path):
    data = np.load(load_path)
    return data['features'], data['labels']
