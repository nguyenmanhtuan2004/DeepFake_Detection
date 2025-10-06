# efficientnet.py
import warnings
import torch
import torch.nn as nn

__all__ = ["EfficientNetB3", "create_model", "EfficientNetB3_pretrained"]

# Các tên model "an toàn" trên nhiều version timm (không bắt buộc tf_*_ns)
_SAFE_CANDIDATES = ("efficientnet_b3", "tf_efficientnet_b3")

def _pick_model_name(timm, preferred: str | None):
    avail = set(timm.list_models())
    if preferred and preferred in avail:
        return preferred
    for name in _SAFE_CANDIDATES:
        if name in avail:
            return name
    raise ValueError(
        f"Không tìm thấy EfficientNet-B3 phù hợp trong timm. "
        f"Đã thử: { (preferred,) if preferred else () } + {_SAFE_CANDIDATES}"
    )

def _get_classifier_module(m: nn.Module) -> nn.Module | None:
    if hasattr(m, "get_classifier"):
        try:
            c = m.get_classifier()
            if isinstance(c, nn.Module): return c
        except Exception:
            pass
    if hasattr(m, "classifier") and isinstance(m.classifier, nn.Module):
        return m.classifier
    return None

class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 (timm) — thân thiện Kaggle/1 GPU.
    - Không yêu cầu 'tf_efficientnet_b3_ns'; ưu tiên 'efficientnet_b3'.
    - Fallback pretrained=False nếu không tải được weights.
    - freeze_backbone: chỉ train head vài epoch đầu.
    """
    def __init__(
        self,
        num_classes: int = 2,
        drop_connect_rate: float = 0.2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str | None = None,   # ép tên cụ thể nếu muốn
    ):
        super().__init__()
        # Đăng ký trước để mọi replica (nếu có) đều có thuộc tính
        self.backbone = nn.Identity()
        self.model_name = model_name or ""

        try:
            import timm
        except ImportError as e:
            raise ImportError("Cần cài timm: pip install timm") from e

        # Chọn tên model an toàn theo version timm
        name = _pick_model_name(timm, preferred=model_name)
        self.model_name = name

        # Tạo model: thử pretrained trước, fail thì về pretrained=False (offline/cache trống)
        try:
            net = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
                drop_path_rate=drop_connect_rate,
                global_pool="avg",
            )
        except Exception as ex:
            if pretrained:
                warnings.warn(
                    f"[EffB3] Không tải được pretrained cho '{name}': {ex}. "
                    "Fallback sang pretrained=False."
                )
                net = timm.create_model(
                    name,
                    pretrained=False,
                    num_classes=num_classes,
                    drop_rate=dropout,
                    drop_path_rate=drop_connect_rate,
                    global_pool="avg",
                )
            else:
                raise

        self.backbone = net

        if freeze_backbone:
            self.freeze_backbone()

        # Thông tin tiện lợi
        try:
            self.num_features = int(self.backbone.get_num_features())
        except Exception:
            clf = _get_classifier_module(self.backbone)
            self.num_features = getattr(clf, "in_features", -1)

    # ---------- freeze / unfreeze ----------
    def freeze_backbone(self):
        head = _get_classifier_module(self.backbone)
        head_ids = set(id(p) for p in head.parameters()) if head is not None else set()
        for n, p in self.backbone.named_parameters():
            p.requires_grad = (id(p) in head_ids) or ("classifier" in n)

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ---------- forward ----------
    def forward(self, x):
        if isinstance(self.backbone, nn.Identity):
            raise RuntimeError("Backbone chưa được gán. Kiểm tra import/khởi tạo model.")
        return self.backbone(x)

# Factories giữ tương thích với trainer cũ
def EfficientNetB3_pretrained(num_classes=2, drop_connect_rate=0.2, dropout=0.3, freeze_backbone=False):
    return EfficientNetB3(
        num_classes=num_classes,
        drop_connect_rate=drop_connect_rate,
        dropout=dropout,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        model_name=None,  # để auto-pick từ _SAFE_CANDIDATES
    )

def create_model(num_classes=2, drop_connect_rate=0.2, dropout=0.3, freeze_backbone=False):
    return EfficientNetB3_pretrained(num_classes, drop_connect_rate, dropout, freeze_backbone)

if __name__ == "__main__":
    # Test nhanh (không cần Internet)
    m = EfficientNetB3(num_classes=2, pretrained=False, freeze_backbone=True)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Model:", m.model_name, "| Output:", y.shape)
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in m.parameters())
    print(f"Trainable: {tr/1e6:.2f}M / Total: {tt/1e6:.2f}M")
