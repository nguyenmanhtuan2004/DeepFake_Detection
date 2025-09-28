# efficientnet.py
import torch
import torch.nn as nn

class EfficientNetB3(nn.Module):
    """
    Wrapper EfficientNet-B3 dùng pretrained từ timm (tf_efficientnet_b3_ns)
    - num_classes: số lớp output (2 cho fake/real)
    - drop_connect_rate: drop path (stochastic depth)
    - dropout: dropout ở head
    - pretrained: có tải trọng số ImageNet sẵn không
    - freeze_backbone: nếu True sẽ freeze toàn bộ backbone (trừ classifier)
                       -> phù hợp phase 1: train nhanh phần head vài epoch
    """
    def __init__(self,
                 num_classes: int = 2,
                 drop_connect_rate: float = 0.2,
                 dropout: float = 0.3,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError(
                "Cần cài `timm` để dùng EfficientNet pretrained. Cài: pip install timm"
            ) from e

        # 'tf_efficientnet_b3_ns' = bản B3 tốt, pretrained ImageNet (Noisy Student)
        self.backbone = timm.create_model(
            "tf_efficientnet_b3_ns",
            pretrained=pretrained,
            num_classes=num_classes,           # tự tạo classifier đúng num_classes
            drop_rate=dropout,                 # dropout ở head
            drop_path_rate=drop_connect_rate   # stochastic depth toàn mạng
        )

        if freeze_backbone:
            # Freeze tất cả trừ classifier
            for name, p in self.backbone.named_parameters():
                if "classifier" in name:
                    continue
                p.requires_grad = False

        self._frozen = freeze_backbone

    def unfreeze(self):
        """Gọi hàm này sau vài epoch để fine-tune toàn bộ backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x):
        return self.backbone(x)

# Factory để hợp với trainer hiện tại
def EfficientNetB3_pretrained(num_classes=2,
                              drop_connect_rate=0.2,
                              dropout=0.3,
                              freeze_backbone=False):
    return EfficientNetB3(num_classes=num_classes,
                          drop_connect_rate=drop_connect_rate,
                          dropout=dropout,
                          pretrained=True,
                          freeze_backbone=freeze_backbone)

# Giữ tên create_model cho tương thích nếu trainer gọi
def create_model(num_classes=2, drop_connect_rate=0.2, dropout=0.3, freeze_backbone=False):
    return EfficientNetB3_pretrained(num_classes, drop_connect_rate, dropout, freeze_backbone)

if __name__ == "__main__":
    # Quick test
    model = EfficientNetB3(num_classes=2, drop_connect_rate=0.2, dropout=0.3, pretrained=True, freeze_backbone=True)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print("Output:", y.shape)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")
