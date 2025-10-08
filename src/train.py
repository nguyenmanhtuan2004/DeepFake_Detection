import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data_loader import create_dataloaders
from feature_extractor import extract_and_stack_features, save_features, load_features
from feature_selector import select_features_ensemble
from models import MetaLearnerMLP
import os

class Trainer:
    def __init__(self, data_dir, batch_size=32, device='cuda', k_features=1000):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.device = device
        self.k_features = k_features
        self.feature_dir = 'features'
        os.makedirs(self.feature_dir, exist_ok=True)
        
    def extract_train_features(self):
        train_path = f'{self.feature_dir}/train_features.npz'
        if os.path.exists(train_path):
            X_train, y_train = load_features(train_path)
        else:
            train_loader, _, _ = create_dataloaders(self.data_dir, self.batch_size, img_size=224, num_workers=4)
            X_train, y_train = extract_and_stack_features(train_loader, self.device)
            save_features(X_train, y_train, train_path)
        return X_train, y_train
    
    def extract_eval_features(self, split):
        path = f'{self.feature_dir}/{split}_features.npz'
        if os.path.exists(path):
            return load_features(path)
        _, val_loader, test_loader = create_dataloaders(self.data_dir, self.batch_size, img_size=224, num_workers=4)
        loader = val_loader if split == 'val' else test_loader
        X, y = extract_and_stack_features(loader, self.device)
        save_features(X, y, path)
        return X, y
    
    def train_meta_learner(self, X_train, y_train, X_val, y_val, epochs=200, lr=0.001, batch_size=256):
        input_dim = X_train.shape[1]
        model = MetaLearnerMLP(input_dim=input_dim).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
        
        # Create DataLoader cho mini-batch training
        from torch.utils.data import TensorDataset, DataLoader
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )
        
        X_val_tensor = torch.FloatTensor(X_val).to(self.device)
        y_val_tensor = torch.LongTensor(y_val).to(self.device)
        
        best_val_acc = 0
        best_model_state = None
        
        for epoch in range(epochs):
            # Training với mini-batches
            model.train()
            train_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * X_batch.size(0)
            
            train_loss /= len(X_train)
            
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_tensor)
                val_loss = criterion(val_outputs, y_val_tensor)
                val_preds = val_outputs.argmax(dim=1).cpu().numpy()
                val_acc = accuracy_score(y_val, val_preds)
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_model_state = model.state_dict().copy()
                
                scheduler.step(val_acc)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss.item():.4f} - Val Acc: {val_acc:.4f} - Best: {best_val_acc:.4f}')
        
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        return model
    
    def evaluate(self, model, X_test, y_test):
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test).to(self.device)
        
        with torch.no_grad():
            outputs = model(X_test_tensor)
            preds = outputs.argmax(dim=1).cpu().numpy()
        
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        
        print(f'\nTest Results:')
        print(f'Accuracy:  {acc:.4f}')
        print(f'Precision: {prec:.4f}')
        print(f'Recall:    {rec:.4f}')
        print(f'F1-Score:  {f1:.4f}')
        
        return acc, prec, rec, f1
    
    def run(self):
        from sklearn.preprocessing import StandardScaler
        
        print("Step 1: Extract Train Features")
        X_train, y_train = self.extract_train_features()
        print(f"Train: {X_train.shape}")
        
        print("\nStep 2: Normalize Features")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        
        print("\nStep 3: Extract Val Features")
        X_val, y_val = self.extract_eval_features('val')
        X_val = scaler.transform(X_val)
        print(f"Val: {X_val.shape}")
        
        print("\nStep 4: Feature Selection")
        indices = select_features_ensemble(X_train, y_train, k=self.k_features)
        X_train_sel = X_train[:, indices]
        X_val_sel = X_val[:, indices]
        print(f"Selected: {X_train_sel.shape}")
        
        print("\nStep 5: Train Meta-Learner")
        model = self.train_meta_learner(X_train_sel, y_train, X_val_sel, y_val, epochs=100, lr=0.001, batch_size=512)
        
        print("\nStep 6: Extract Test Features & Evaluate")
        X_test, y_test = self.extract_eval_features('test')
        X_test = scaler.transform(X_test)
        X_test_sel = X_test[:, indices]
        print(f"Test: {X_test_sel.shape}")
        self.evaluate(model, X_test_sel, y_test)

if __name__ == '__main__':
    trainer = Trainer(
        data_dir='../Dataset',
        batch_size=32,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        k_features=2000
    )
    trainer.run()
