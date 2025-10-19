"""
Data Loader cho Deepfake Detection
Hỗ trợ cấu trúc: Dataset/{train,val,test}/{fake,real}/*.jpg
"""

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
from pathlib import Path


class DeepfakeDataset(Dataset):
    """Dataset cho Deepfake Detection"""
    
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: đường dẫn đến folder Dataset
            split: 'train', 'val', hoặc 'test'
            transform: các phép biến đổi ảnh
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Lấy danh sách ảnh và label
        self.samples = []
        self._load_samples()
        
    def _load_samples(self):
        """Load tất cả ảnh từ fake và real folders"""
        split_dir = self.root_dir / self.split
        
        # Label: 0 = real, 1 = fake
        for label, class_name in enumerate(['real', 'fake']):
            class_dir = split_dir / class_name
            if not class_dir.exists():
                print(f"Warning: {class_dir} không tồn tại!")
                continue
                
            # Lấy tất cả file ảnh
            for img_path in class_dir.glob('*.jpg'):
                self.samples.append((str(img_path), label))
        
        print(f"Loaded {len(self.samples)} samples from {split_dir}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        # Load ảnh
        image = Image.open(img_path).convert('RGB')
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
            
        return image, label


def get_transforms(img_size=224, augment=True):
    mean, std = [0.485,0.456,0.406], [0.229,0.224,0.225]
    base_resize = transforms.Resize(int(img_size*1.15), interpolation=transforms.InterpolationMode.BILINEAR)
    if augment:
        # Transform cho training (có augmentation)
        transform = transforms.Compose([
            base_resize,
            transforms.RandomCrop(img_size),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
        ])
    else:
        # Transform cho validation/test (không augmentation)
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    return transform


def create_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4):
    """
    Tạo DataLoaders cho train, val, và test
    
    Args:
        data_dir: đường dẫn đến folder Dataset
        batch_size: kích thước batch
        img_size: kích thước ảnh (224 cho EfficientNet-B3)
        num_workers: số workers cho DataLoader
    
    Returns:
        train_loader, val_loader, test_loader
    """
    # Tạo transforms
    train_transform = get_transforms(img_size=img_size, augment=True)
    val_transform = get_transforms(img_size=img_size, augment=False)
    
    # Tạo datasets
    train_dataset = DeepfakeDataset(data_dir, split='train', transform=train_transform)
    val_dataset = DeepfakeDataset(data_dir, split='val', transform=val_transform)
    test_dataset = DeepfakeDataset(data_dir, split='test', transform=val_transform)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# Test nhanh
if __name__ == "__main__":
    # Test dataset
    data_dir = "../Dataset"
    
    print("=" * 50)
    print("Testing DataLoader...")
    print("=" * 50)
    
    train_loader, val_loader, test_loader = create_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        img_size=224,
        num_workers=0  # 0 cho test nhanh
    )
    
    # Kiểm tra 1 batch
    images, labels = next(iter(train_loader))
    print(f"\nTrain batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels shape: {labels.shape}")
    print(f"  Labels: {labels}")
    print(f"  Real: {(labels == 0).sum()}, Fake: {(labels == 1).sum()}")
    
    images, labels = next(iter(val_loader))
    print(f"\nVal batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels: {labels}")
    
    images, labels = next(iter(test_loader))
    print(f"\nTest batch:")
    print(f"  Images shape: {images.shape}")
    print(f"  Labels: {labels}")
    
    print("\n" + "=" * 50)
    print("DataLoader hoạt động tốt! ✓")
    print("=" * 50)
