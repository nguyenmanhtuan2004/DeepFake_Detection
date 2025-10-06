# Deepfake Detection

Implementation of **Feature Stacking + Meta-Learning** for deepfake detection based on [this paper](https://www.sciencedirect.com/science/article/pii/S2405844024019649).

## Pipeline

```
Images → Xception + EfficientNet-B3 → Feature Stacking → Feature Selection (RF+XGBoost) → MLP Meta-Learner
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train baseline models (optional)
cd baseline
jupyter notebook train.ipynb

# Train meta-learning pipeline
cd src
python train.py
```

## Project Structure

```
DeepFake_Detection/
├── baseline/                   # Individual model training
│   ├── train.ipynb            # Train Xception/EfficientNet-B3
│   ├── my_xception.py         # Xception model
│   └── my_efficientnet.py     # EfficientNet-B3 model
├── src/                       # Meta-learning pipeline
│   ├── data_loader.py         # Dataset loader
│   ├── models.py              # Feature extractors + MLP
│   ├── feature_extractor.py   # Extract & stack features
│   ├── feature_selector.py    # RF + XGBoost selection
│   └── train.py              # Main training script
└── Dataset/                   # Data structure
    ├── train/{real,fake}/
    ├── val/{real,fake}/
    └── test/{real,fake}/
```

## Method

1. **Feature Extraction**: Xception (2048) + EfficientNet-B3 (1536) = 3584 features
2. **Feature Stacking**: Concatenate features from both models
3. **Feature Selection**: Ensemble ranking with RF + XGBoost → select top 1000 features
4. **Meta-Learning**: Train MLP (1000 → 512 → 256 → 2) on selected features

## Results

Paper achieves:
- **Celeb-DF (V2)**: 96.33% accuracy
- **FaceForensics++**: 98.00% accuracy

## Citation

```bibtex
@article{naskar2024deepfake,
  title={Deepfake detection using deep feature stacking and meta-learning},
  author={Naskar, Gourab and Mohiuddin, Sk and Malakar, Samir and Cuevas, Erik and Sarkar, Ram},
  journal={Heliyon},
  volume={10},
  number={4},
  year={2024}
}
```