import torch
import numpy as np
from data_loader import create_dataloaders
from feature_extractor import extract_and_stack_features, load_features
from feature_selector import select_features_rf, select_features_xgboost, select_features_ensemble
from models import MetaLearnerMLP
from sklearn.metrics import accuracy_score
import os

def quick_train_eval(X_train, y_train, X_val, y_val, device='cuda', epochs=30):
    input_dim = X_train.shape[1]
    model = MetaLearnerMLP(input_dim=input_dim).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    X_train_t = torch.FloatTensor(X_train).to(device)
    y_train_t = torch.LongTensor(y_train).to(device)
    X_val_t = torch.FloatTensor(X_val).to(device)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_t)
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_t)
        val_preds = val_outputs.argmax(dim=1).cpu().numpy()
        val_acc = accuracy_score(y_val, val_preds)
    
    return val_acc

def compare_methods(data_dir='../Dataset', k=1000):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if os.path.exists('features/train_features.npz'):
        print("Loading features...")
        X_train, y_train = load_features('features/train_features.npz')
        X_val, y_val = load_features('features/val_features.npz')
        X_test, y_test = load_features('features/test_features.npz')
    else:
        print("Extract features first by running train.py")
        return
    
    print(f"\nComparing Feature Selection Methods (k={k})")
    print("=" * 60)
    
    print("\n1. Random Forest")
    X_train_rf, X_val_rf, X_test_rf, _ = select_features_rf(X_train, y_train, X_val, X_test, k=k)
    acc_rf = quick_train_eval(X_train_rf, y_train, X_val_rf, y_val, device)
    print(f"   Validation Accuracy: {acc_rf:.4f}")
    
    print("\n2. XGBoost")
    X_train_xgb, X_val_xgb, X_test_xgb, _ = select_features_xgboost(X_train, y_train, X_val, X_test, k=k)
    acc_xgb = quick_train_eval(X_train_xgb, y_train, X_val_xgb, y_val, device)
    print(f"   Validation Accuracy: {acc_xgb:.4f}")
    
    print("\n3. Ensemble (RF + XGBoost) - PAPER METHOD")
    X_train_ens, X_val_ens, X_test_ens, _ = select_features_ensemble(X_train, y_train, X_val, X_test, k=k)
    acc_ens = quick_train_eval(X_train_ens, y_train, X_val_ens, y_val, device)
    print(f"   Validation Accuracy: {acc_ens:.4f}")
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    print(f"Random Forest:     {acc_rf:.4f}")
    print(f"XGBoost:           {acc_xgb:.4f}")
    print(f"Ensemble (Paper):  {acc_ens:.4f} ⭐")
    print("=" * 60)
    
    best_method = max([('RF', acc_rf), ('XGB', acc_xgb), ('Ensemble', acc_ens)], key=lambda x: x[1])
    print(f"\n🏆 Best Method: {best_method[0]} with {best_method[1]:.4f} accuracy")

if __name__ == '__main__':
    compare_methods(k=1000)
