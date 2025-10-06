# So sánh Feature Selection Methods

## Paper Method vs Others

| Method | Description | Pros | Cons | Paper Use |
|--------|-------------|------|------|-----------|
| **Random Forest** | Feature importance từ RF trees | - Fast<br>- Handle non-linear<br>- Robust | - Bias với high cardinality | ✅ YES |
| **XGBoost** | Gradient boosting importance | - Very accurate<br>- Handle imbalance | - Slower training | ✅ YES |
| **Ensemble (RF+XGB)** | Average importance từ cả 2 | - **Best of both**<br>- More stable<br>- Reduce bias | - Need train 2 models | ✅✅ PAPER |
| Mutual Information | Statistical dependency | - Model-agnostic<br>- Fast | - Linear assumption | ❌ NO |
| ANOVA F-test | Statistical test | - Fast<br>- Interpretable | - Linear only | ❌ NO |

## Tại sao Paper dùng RF + XGBoost?

### 1. **Complementary Strengths**
- **RF**: Captures feature interactions through bagging
- **XGBoost**: Captures sequential patterns through boosting
- **Ensemble**: Combines both perspectives

### 2. **Tree-based Importance**
```
Feature Importance = Weighted average của:
- Số lần feature được dùng để split
- Information gain từ mỗi split
- Position trong tree (root → leaf)
```

### 3. **Ranking-based Selection**
```python
# RF importance
rf_scores = [0.05, 0.12, 0.08, ...]  # 3584 features

# XGBoost importance  
xgb_scores = [0.07, 0.10, 0.09, ...]

# Ensemble
ensemble_scores = (rf_scores + xgb_scores) / 2

# Select top K
top_k_indices = argsort(ensemble_scores)[-1000:]
```

## Implementation Comparison

### ❌ Cách cũ (Statistical):
```python
# Mutual Information - không theo paper
selector = SelectKBest(mutual_info_classif, k=1000)
X_selected = selector.fit_transform(X, y)
```

### ✅ Cách paper (Tree-based Ensemble):
```python
# Train RF
rf = RandomForestClassifier(n_estimators=100)
rf.fit(X_train, y_train)

# Train XGBoost
xgb = XGBClassifier(n_estimators=100)
xgb.fit(X_train, y_train)

# Combine importances
importances = (rf.feature_importances_ + xgb.feature_importances_) / 2

# Select top K
indices = argsort(importances)[::-1][:k]
X_selected = X[:, indices]
```

## Expected Results

Paper đạt được:
- Celeb-DF: **96.33%** accuracy
- FaceForensics++: **98.00%** accuracy

Với feature selection đúng method này, kỳ vọng tăng **1-3%** so với statistical methods!

## Kết luận

✅ **Cách của paper hay hơn** vì:
1. Tree-based models handle non-linear relationships tốt hơn
2. Ensemble giảm bias và variance
3. Phù hợp với deep features (high-dimensional, non-linear)
4. Đã được validate với kết quả cao trong paper

Code hiện tại đã implement đúng theo paper! 🎯
