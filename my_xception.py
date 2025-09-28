# xception.py  — safe 1-GPU pretrained wrapper for timm
import warnings
import torch
import torch.nn as nn

__all__ = ["Xception", "create_model", "Xception_pretrained"]

# Ưu tiên các tên có mặt ở nhiều version timm (không bắt buộc tf_*):
_SAFE_CANDIDATES = ("xception", "tf_xception", "xception41", "xception65", "xception71")
_HEAD_KEYS = ("classifier", "fc", "head", "last_linear")

def _pick_model_name(timm, preferred: str | None):
    avail = set(timm.list_models())
    if preferred and preferred in avail:
        return preferred
    for name in _SAFE_CANDIDATES:
        if name in avail:
            return name
    raise ValueError(
        f"Không tìm thấy Xception phù hợp trong timm. "
        f"Đã thử: { (preferred,) if preferred else () } + {_SAFE_CANDIDATES}"
    )

def _get_classifier_module(m: nn.Module) -> nn.Module | None:
    if hasattr(m, "get_classifier"):
        try:
            c = m.get_classifier()
            if isinstance(c, nn.Module):
                return c
        except Exception:
            pass
    # timm xception thường dùng 'fc' làm head
    for k in _HEAD_KEYS:
        if hasattr(m, k) and isinstance(getattr(m, k), nn.Module):
            return getattr(m, k)
    return None

class Xception(nn.Module):
    """
    Xception (timm) — an toàn cho 1 GPU.
      - num_classes: số lớp output
      - drop_rate: dropout ở head
      - drop_path_rate: stochastic depth (nếu model hỗ trợ)
      - pretrained: dùng trọng số ImageNet nếu có cache/Internet
      - freeze_backbone: chỉ train head vài epoch đầu
      - model_name: ép tên (vd 'xception'); để None sẽ auto-pick từ _SAFE_CANDIDATES
    """
    def __init__(
        self,
        num_classes: int = 2,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_name: str | None = None,
    ):
        super().__init__()
        # Đăng ký trước để luôn có thuộc tính (dù chỉ 1 GPU, vẫn an toàn):
        self.backbone = nn.Identity()
        self.model_name = model_name or ""

        try:
            import timm
        except ImportError as e:
            raise ImportError("Cần cài timm: pip install timm") from e

        name = _pick_model_name(timm, preferred=model_name)
        self.model_name = name

        # Tạo model: thử pretrained trước; fail thì fallback về pretrained=False
        try:
            net = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                global_pool="avg",
            )
        except Exception as ex:
            if pretrained:
                warnings.warn(
                    f"[Xception] Không tải được pretrained cho '{name}': {ex}. "
                    "Fallback sang pretrained=False."
                )
                net = timm.create_model(
                    name,
                    pretrained=False,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate,
                    global_pool="avg",
                )
            else:
                raise

        self.backbone = net

        if freeze_backbone:
            self.freeze_backbone()

        # Thông tin tiện ích (không bắt buộc)
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
            # giữ head trainable
            p.requires_grad = (id(p) in head_ids) or any(k in n for k in _HEAD_KEYS)

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    # ---------- forward ----------
    def forward(self, x):
        if isinstance(self.backbone, nn.Identity):
            raise RuntimeError("Backbone chưa được gán. Kiểm tra import/khởi tạo model.")
        return self.backbone(x)

# Factories giữ tương thích với trainer cũ
def Xception_pretrained(num_classes=2, drop_rate=0.0, drop_path_rate=0.0, freeze_backbone=False):
    return Xception(
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
        pretrained=True,
        freeze_backbone=freeze_backbone,
        model_name=None,  # auto-pick an toàn
    )

def create_model(num_classes=2, drop_rate=0.0, drop_path_rate=0.0, freeze_backbone=False):
    return Xception_pretrained(num_classes, drop_rate, drop_path_rate, freeze_backbone)

if __name__ == "__main__":
    # Test nhanh (offline)
    m = Xception(num_classes=2, pretrained=False, freeze_backbone=True)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Model:", m.model_name, "| Out:", y.shape)
    tr = sum(p.numel() for p in m.parameters() if p.requires_grad)
    tt = sum(p.numel() for p in m.parameters())
    print(f"Trainable: {tr/1e6:.2f}M / Total: {tt/1e6:.2f}M")
