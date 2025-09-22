# EfficientNet-B7 for DeepFake Detection

A comprehensive implementation of EfficientNet-B7 for binary classification of real vs fake images in deepfake detection tasks.

## Features

- **EfficientNet-B7 Architecture**: Full implementation with proper scaling parameters
- **Training Pipeline**: Complete training script with data augmentation and validation
- **Evaluation Tools**: Comprehensive evaluation with multiple metrics and visualizations
- **Data Preprocessing**: Utilities for data cleaning, resizing, and augmentation
- **Inference Engine**: Real-time inference for single images or batch processing

## Project Structure

```
DeepFake_Detection/
├── efficientnet_b7.py          # EfficientNet-B7 model implementation
├── train_efficientnet_b7.py    # Training script
├── evaluate_model.py           # Evaluation and testing script
├── inference.py                # Inference script for new images
├── data_utils.py               # Data preprocessing utilities
├── requirements.txt            # Python dependencies
└── README.md                   # This file

Dataset/                        # Your dataset structure
├── train/
│   ├── real/
│   └── fake/
├── val/
│   ├── real/
│   └── fake/
└── test/
    ├── real/
    └── fake/
```

## Installation

1. **Clone or download** this repository
2. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

### Required Dependencies

- Python 3.8+
- PyTorch 2.0+
- torchvision
- NumPy
- Pillow (PIL)
- OpenCV
- scikit-learn
- matplotlib
- seaborn

## Usage

### 1. Data Preparation

First, organize your data in the following structure:
```
Dataset/
├── train/
│   ├── real/     # Real images for training
│   └── fake/     # Fake images for training
├── val/
│   ├── real/     # Real images for validation
│   └── fake/     # Fake images for validation
└── test/
    ├── real/     # Real images for testing
    └── fake/     # Fake images for testing
```

Use the data preprocessing utilities:
```powershell
python data_utils.py
```

This will:
- Analyze your dataset statistics
- Clean corrupted images
- Optionally resize images to optimal size
- Balance classes through augmentation

### 2. Training

Train the EfficientNet-B7 model:
```powershell
python train_efficientnet_b7.py
```

**Training Configuration:**
- Input size: 380x380 (optimized for memory efficiency)
- Batch size: 8 (adjust based on your GPU memory)
- Learning rate: 0.0001 with cosine annealing
- Epochs: 30 (with early stopping)
- Optimizer: AdamW with weight decay

**Training Features:**
- Data augmentation (rotation, flip, color jitter, etc.)
- Stochastic depth for regularization
- Learning rate scheduling
- Model checkpointing
- Training history visualization

### 3. Evaluation

Evaluate the trained model:
```powershell
python evaluate_model.py
```

This provides:
- **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC
- **Visualizations**: Confusion matrix, ROC curve, probability distributions
- **Misclassification analysis**: Identify and analyze model failures
- **Multi-dataset comparison**: Compare performance across different datasets

### 4. Inference

Use the trained model for prediction:

**Single image:**
```powershell
python inference.py --input path/to/image.jpg --model efficientnet_b7_deepfake.pth
```

**Batch processing:**
```powershell
python inference.py --input path/to/folder/ --model efficientnet_b7_deepfake.pth --output results.csv
```

**Custom threshold:**
```powershell
python inference.py --input image.jpg --model model.pth --threshold 0.7
```

## Model Architecture

### EfficientNet-B7 Specifications

- **Input size**: 600x600 (training), 380x380 (efficient inference)
- **Parameters**: ~66M parameters
- **Width multiplier**: 2.0
- **Depth multiplier**: 3.1
- **Dropout rate**: 0.5

### Key Components

1. **Stem**: Initial convolution + BatchNorm + Swish
2. **MBConv Blocks**: Mobile inverted bottleneck convolution blocks
3. **Squeeze-and-Excitation**: Channel attention mechanism
4. **Stochastic Depth**: Regularization during training
5. **Head**: Final convolution + Global average pooling + Classifier

### Building Blocks

- **Swish Activation**: x * sigmoid(x)
- **Squeeze-and-Excitation**: Channel-wise attention
- **MBConv**: Expansion → Depthwise → SE → Projection
- **Stochastic Depth**: Random skip connections during training

## Configuration Options

### Training Parameters

```python
config = {
    'input_size': 380,      # Image input size
    'batch_size': 8,        # Batch size (adjust for your GPU)
    'num_epochs': 30,       # Training epochs
    'learning_rate': 0.0001, # Initial learning rate
    'dropout_rate': 0.5,    # Dropout in classifier
    'weight_decay': 0.01,   # AdamW weight decay
}
```

### Data Augmentation

- Random horizontal flip (50%)
- Random rotation (±10°)
- Color jitter (brightness, contrast, saturation, hue)
- Random crop with resize
- Gaussian blur (20%)

## Performance Optimization

### Memory Optimization

1. **Reduced input size**: 380x380 instead of 600x600
2. **Gradient checkpointing**: Enable for very large models
3. **Mixed precision**: Use `torch.cuda.amp` for faster training
4. **Batch size adjustment**: Start with smaller batches

### Speed Optimization

1. **Model compilation**: Use `torch.compile()` in PyTorch 2.0+
2. **Data loading**: Increase `num_workers` based on CPU cores
3. **Pin memory**: Enable for GPU training

## Expected Results

### Performance Metrics

On typical deepfake datasets:
- **Accuracy**: 85-95%
- **Precision**: 80-90%
- **Recall**: 85-95%
- **F1-Score**: 82-92%
- **AUC-ROC**: 0.90-0.98

### Training Time

- **Single GPU (RTX 3080)**: ~2-4 hours for 30 epochs
- **CPU only**: ~12-24 hours (not recommended)

## Troubleshooting

### Common Issues

1. **CUDA out of memory**:
   - Reduce batch size to 4 or 2
   - Reduce input size to 224x224
   - Enable gradient checkpointing

2. **Slow training**:
   - Increase `num_workers` in DataLoader
   - Enable pin_memory for GPU training
   - Use mixed precision training

3. **Poor performance**:
   - Check data quality and balance
   - Increase training epochs
   - Adjust learning rate
   - Add more data augmentation

### Dataset Requirements

- **Minimum**: 1000 images per class
- **Recommended**: 5000+ images per class
- **Image quality**: High resolution, clear faces
- **Balance**: Similar number of real and fake images

## Model Variants

You can easily modify the model for different use cases:

### Different Input Sizes
```python
model = create_efficientnet_b7(num_classes=2)
# Modify input size in training config
```

### Multi-class Classification
```python
model = create_efficientnet_b7(num_classes=4)  # For 4 classes
```

### Feature Extraction
```python
features = model.extract_features(x)  # Get feature maps
feature_maps = model.get_feature_maps(x)  # Get intermediate features
```

## Citation

If you use this implementation in your research, please cite:

```bibtex
@article{tan2019efficientnet,
  title={EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks},
  author={Tan, Mingxing and Le, Quoc V},
  journal={arXiv preprint arXiv:1905.11946},
  year={2019}
}
```

## License

This project is for educational and research purposes. Please check the licenses of the dependencies and datasets you use.

## Contributing

Feel free to submit issues, feature requests, or improvements to this implementation.

---

**Note**: This implementation is optimized for deepfake detection but can be adapted for other binary classification tasks by modifying the dataset structure and class names.