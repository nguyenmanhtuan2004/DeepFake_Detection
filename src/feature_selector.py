import numpy as np
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def select_features_rf(X_train, y_train, X_val, X_test, k=1000):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    
    X_train_selected = X_train[:, indices]
    X_val_selected = X_val[:, indices]
    X_test_selected = X_test[:, indices]
    
    print(f"RF: Selected {k} features from {X_train.shape[1]} features")
    return X_train_selected, X_val_selected, X_test_selected, indices

def select_features_xgboost(X_train, y_train, X_val, X_test, k=1000):
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    xgb.fit(X_train, y_train)
    
    importances = xgb.feature_importances_
    indices = np.argsort(importances)[::-1][:k]
    
    X_train_selected = X_train[:, indices]
    X_val_selected = X_val[:, indices]
    X_test_selected = X_test[:, indices]
    
    print(f"XGBoost: Selected {k} features from {X_train.shape[1]} features")
    return X_train_selected, X_val_selected, X_test_selected, indices

def select_features_ensemble(X_train, y_train, k=1000):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    xgb = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
    
    rf.fit(X_train, y_train)
    xgb.fit(X_train, y_train)
    
    combined_importances = (rf.feature_importances_ + xgb.feature_importances_) / 2
    indices = np.argsort(combined_importances)[::-1][:k]
    
    return indices
