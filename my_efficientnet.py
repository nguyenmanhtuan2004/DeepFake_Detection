# efficientnet.py
import warnings
import torch
import torch.nn as nn

_HEAD_KEYS = ("classifier", "fc", "head", "last_linear")

class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 pretrained (timm, tf_efficientnet_b3_ns).
    - num_classes: số lớp output
    - drop_connect_rate: stochastic depth
    - dropout: dropout ở head
    - pretrained: dùng trọng số ImageNet (tải từ cache/online)
    - freeze_backbone: freeze mọi thứ trừ head trong vài epoch đầu
    """
    def __init__(
        self,
        num_classes: int = 2,
        drop_connect_rate: float = 0.2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Cần cài timm: pip install timm") from e

        # Tạo model; nếu pretrained không tải được (không có internet/cache), fallback sang random init
        try:
            self.backbone = timm.create_model(
                "efficientnet_b3",
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
                drop_path_rate=drop_connect_rate,
                global_pool="avg",
            )
        except Exception as ex:
            if pretrained:
                warnings.warn(
                    f"[EfficientNetB3] Không tải được pretrained weights ({ex}). "
                    "Đang fallback sang pretrained=False."
                )
                self.backbone = timm.create_model(
                    "tf_efficientnet_b3",
                    pretrained=False,
                    num_classes=num_classes,
                    drop_rate=dropout,
                    drop_path_rate=drop_connect_rate,
                    global_pool="avg",
                )
            else:
                raise

        if freeze_backbone:
            self._freeze_all_but_head()

        self._frozen = freeze_backbone

    # ---- tiện ích freeze/unfreeze ----
    def _is_head_param(self, name: str) -> bool:
        return any(k in name for k in _HEAD_KEYS)

    def _freeze_all_but_head(self):
        for n, p in self.backbone.named_parameters():
            if not self._is_head_param(n):
                p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x):
        return self.backbone(x)


# Factory giữ tương thích với trainer cũ
def EfficientNetB3_pretrained(num_classes=2, drop_connect_rate=0.2, dropout=0.3, freeze_backbone=False):
    return EfficientNetB3(
        num_classes=num_classes,
        drop_connect_rate=drop_connect_rate,
        dropout=dropout,
        pretrained=True,
        freeze_backbone=freeze_backbone,
    )

def create_model(num_classes=2, drop_connect_rate=0.2, dropout=0.3, freeze_backbone=False):
    return EfficientNetB3_pretrained(num_classes, drop_connect_rate, dropout, freeze_backbone)


if __name__ == "__main__":
    # Quick test: dùng pretrained=False để không cần internet
    m = EfficientNetB3(num_classes=2, drop_connect_rate=0.2, dropout=0.3, pretrained=False, freeze_backbone=True)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Output:", y.shape)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")
