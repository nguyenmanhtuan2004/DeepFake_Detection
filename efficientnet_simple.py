import torch
import torch.nn as nn
import math


class Swish(nn.Module):
    """Swish activation function"""
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            Swish(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck Convolution with Stochastic Depth"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, drop_rate=0.0):
        super().__init__()
        self.stride = stride
        self.use_skip = stride == 1 and in_channels == out_channels
        self.drop_rate = drop_rate
        
        # Expand
        hidden_dim = in_channels * expand_ratio
        if expand_ratio != 1:
            self.expand = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
                Swish()
            )
        else:
            self.expand = nn.Identity()
        
        # Depthwise with proper padding
        self.depthwise = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                     kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim, eps=1e-3, momentum=0.01),
            Swish()
        )
        
        # SE block with ratio 0.25
        self.se = SEBlock(hidden_dim, reduction=4)
        
        # Project
        self.project = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-3, momentum=0.01)
        )
    
    def forward(self, x):
        identity = x
        
        # Expand
        x = self.expand(x)
        
        # Depthwise + SE
        x = self.depthwise(x)
        x = self.se(x)
        
        # Project
        x = self.project(x)
        
        # Stochastic depth (drop path)
        if self.use_skip:
            if self.training and self.drop_rate > 0:
                # Random drop with probability drop_rate
                if torch.rand(1) < self.drop_rate:
                    return identity
            x = x + identity
        
        return x


class EfficientNetB7(nn.Module):
    """EfficientNet-B7 Model (Chuẩn theo paper)"""
    
    def __init__(self, num_classes=2, drop_connect_rate=0.2):
        super().__init__()
        self.drop_connect_rate = drop_connect_rate
        
        # EfficientNet-B7 config: [expand_ratio, channels, repeats, stride, kernel_size]
        # Đây là config chuẩn từ paper
        b7_config = [
            [1,  32,  4, 1, 3],   # Stage 1: MBConv1, k3x3
            [6,  48,  7, 2, 3],   # Stage 2: MBConv6, k3x3  
            [6,  80,  7, 2, 5],   # Stage 3: MBConv6, k5x5
            [6, 160, 10, 2, 3],   # Stage 4: MBConv6, k3x3
            [6, 224, 10, 1, 5],   # Stage 5: MBConv6, k5x5
            [6, 384, 13, 2, 5],   # Stage 6: MBConv6, k5x5
            [6, 640,  4, 1, 3],   # Stage 7: MBConv6, k3x3
        ]
        
        # Stem: 3 -> 64 channels
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64, eps=1e-3, momentum=0.01),
            Swish()
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        in_channels = 64
        total_blocks = sum([repeats for _, _, repeats, _, _ in b7_config])
        block_idx = 0
        
        for expand_ratio, out_channels, repeats, stride, kernel_size in b7_config:
            stage_blocks = []
            
            for i in range(repeats):
                # Calculate drop rate for this block (linear increase)
                drop_rate = self.drop_connect_rate * block_idx / total_blocks
                
                # First block in stage uses stride, others use stride=1
                block_stride = stride if i == 0 else 1
                input_channels = in_channels if i == 0 else out_channels
                
                stage_blocks.append(
                    MBConv(
                        input_channels, out_channels, kernel_size, 
                        block_stride, expand_ratio, drop_rate
                    )
                )
                block_idx += 1
            
            self.stages.append(nn.Sequential(*stage_blocks))
            in_channels = out_channels
        
        # Head: 640 -> 2560 -> num_classes
        self.head = nn.Sequential(
            nn.Conv2d(640, 2560, 1, bias=False),
            nn.BatchNorm2d(2560, eps=1e-3, momentum=0.01),
            Swish(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(2560, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_stage(self, in_channels, out_channels, repeats, stride, kernel, expand):
        """Create a stage with multiple MBConv blocks"""
        layers = []
        
        # First block with stride
        layers.append(MBConv(in_channels, out_channels, kernel, stride, expand))
        
        # Remaining blocks
        for _ in range(repeats - 1):
            layers.append(MBConv(out_channels, out_channels, kernel, 1, expand))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Stem
        x = self.stem(x)
        
        # Stages
        for stage in self.stages:
            x = stage(x)
        
        # Head
        x = self.head(x)
        
        return x


def create_model(num_classes=2):
    """Create EfficientNet-B7 model"""
    return EfficientNetB7(num_classes)


# Test model
if __name__ == "__main__":
    model = create_model(num_classes=2)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 224, 224)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")