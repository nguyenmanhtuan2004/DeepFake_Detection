# Deepfake Detection - Feature Stacking + Meta-Learning

Implementation theo paper: "Deepfake detection using deep feature stacking and meta-learning"

## Pipeline

```
Images → Xception + EfficientNet-B3 → Feature Stacking → Feature Selection → MLP Meta-Learner → Prediction
```

## Cấu trúc file

```
src/
├── data_loader.py          # Load dữ liệu từ Dataset
├── models.py               # Xception, EfficientNet-B3, MLP
├── feature_extractor.py    # Trích xuất và stack features
├── feature_selector.py     # Chọn k features tốt nhất
└── train.py               # Train pipeline hoàn chỉnh
```

## Sử dụng

```bash
cd src
python train.py
```

## Các bước trong pipeline

1. **Extract Features**: Trích xuất từ Xception (2048) + EfficientNet-B3 (1536) = 3584 features
2. **Feature Stacking**: Ghép 2 feature vectors
3. **Feature Selection**: Ranking-based với RF + XGBoost ensemble (theo paper)
4. **Meta-Learning**: Train MLP với selected features

## Feature Selection Methods

Paper sử dụng **ensemble ranking** từ Random Forest và XGBoost:
- RF: Feature importance từ tree-based model
- XGBoost: Gradient boosting feature importance
- Ensemble: Trung bình của RF + XGBoost importance

So sánh các phương pháp:
```bash
python compare_feature_selection.py
```

## Hyperparameters

- Batch size: 32
- Image size: 300x300 (EfficientNet-B3)
- K features: 1000
- MLP hidden: 512 → 256 → 2
- Learning rate: 0.001
- Epochs: 50
