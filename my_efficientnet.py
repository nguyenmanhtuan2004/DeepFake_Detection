# efficientnet.py
import warnings
import torch
import torch.nn as nn

_HEAD_KEYS = ("classifier", "fc", "head", "last_linear")

class EfficientNetB3(nn.Module):
    """
    EfficientNet-B3 pretrained qua timm.
    - Tự chọn mô hình khả dụng theo thứ tự ưu tiên:
        'tf_efficientnet_b3_ns' -> 'tf_efficientnet_b3' -> 'efficientnet_b3'
    - Fallback an toàn nếu không tải được pretrained weights (offline / không có trên version timm).
    - freeze_backbone: đóng băng mọi thứ trừ head (classifier) trong vài epoch đầu.
    """
    def __init__(
        self,
        num_classes: int = 2,
        drop_connect_rate: float = 0.2,
        dropout: float = 0.3,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        model_candidates = ("tf_efficientnet_b3", "efficientnet_b3"),
    ):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Cần cài timm: pip install timm") from e

        # 1) Chọn tên model timm khả dụng
        chosen = None
        available = set(timm.list_models())
        for name in model_candidates:
            if name in available:
                chosen = name
                break
        if chosen is None:
            raise ValueError(
                f"Không tìm thấy EfficientNet-B3 trong timm. "
                f"Đã thử: {model_candidates}. Version timm của bạn: {getattr(timm, '__version__', 'unknown')}"
            )

        self.model_name = chosen

        # 2) Tạo backbone, fallback an toàn nếu không tải được weights
        self.backbone = None  # Khởi tạo mặc định
        
        try:
            print(f"[EfficientNetB3] Đang tạo model '{self.model_name}' với pretrained={pretrained}")
            self.backbone = timm.create_model(
                self.model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
                drop_path_rate=drop_connect_rate,
                global_pool="avg",
            )
            print(f"[EfficientNetB3] ✅ Tạo thành công!")
        except Exception as ex:
            print(f"[EfficientNetB3] ❌ Lỗi tạo model: {ex}")
            if pretrained:
                print("[EfficientNetB3] 🔄 Thử fallback với pretrained=False...")
                try:
                    self.backbone = timm.create_model(
                        self.model_name,
                        pretrained=False,
                        num_classes=num_classes,
                        drop_rate=dropout,
                        drop_path_rate=drop_connect_rate,
                        global_pool="avg",
                    )
                    print(f"[EfficientNetB3] ✅ Fallback thành công!")
                    warnings.warn(f"Không tải được pretrained weights, sử dụng random weights.")
                except Exception as ex2:
                    print(f"[EfficientNetB3] ❌ Fallback cũng thất bại: {ex2}")
                    raise RuntimeError(f"Không thể tạo EfficientNetB3 với '{self.model_name}'. "
                                     f"Lỗi gốc: {ex}. Lỗi fallback: {ex2}") from ex2
            else:
                raise RuntimeError(f"Không thể tạo EfficientNetB3 với '{self.model_name}': {ex}") from ex
        
        # Đảm bảo backbone đã được tạo
        if self.backbone is None:
            raise RuntimeError("EfficientNetB3 backbone vẫn là None sau khi khởi tạo!")

        if freeze_backbone:
            self._freeze_all_but_head()

        self._frozen = freeze_backbone

        # (tuỳ chọn) thuộc tính tiện lợi
        self.num_features = self._get_num_features_safely()

    # ----------------- helpers -----------------
    def _get_num_features_safely(self) -> int:
        # timm model nào cũng có get_num_features()
        try:
            return int(self.backbone.get_num_features())
        except Exception:
            # dự phòng: suy ra từ classifier in_features
            clf = self._get_classifier_module()
            if isinstance(clf, nn.Linear):
                return clf.in_features
            return -1

    def _get_classifier_module(self):
        # ưu tiên API timm
        if hasattr(self.backbone, "get_classifier"):
            try:
                clf = self.backbone.get_classifier()
                if isinstance(clf, nn.Module):
                    return clf
            except Exception:
                pass
        # fallback theo tên quen thuộc
        for k in _HEAD_KEYS:
            if hasattr(self.backbone, k) and isinstance(getattr(self.backbone, k), nn.Module):
                return getattr(self.backbone, k)
        return None

    def _is_head_param(self, name: str) -> bool:
        if any(k in name for k in _HEAD_KEYS):
            return True
        # tên head ở một số bản timm có thể là 'classifier'
        return False

    def _freeze_all_but_head(self):
        head = self._get_classifier_module()
        head_ids = set()
        if head is not None:
            head_ids = {id(p) for p in head.parameters()}
        for n, p in self.backbone.named_parameters():
            # giữ head trainable
            if id(p) in head_ids or self._is_head_param(n):
                p.requires_grad = True
            else:
                p.requires_grad = False

    def unfreeze(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    # ----------------- nn.Module -----------------
    def forward(self, x):
        # bảo vệ: nếu vì lý do gì backbone chưa có, báo lỗi rõ ràng
        if not hasattr(self, "backbone"):
            raise RuntimeError("EfficientNetB3 không có thuộc tính 'backbone'. Lỗi trong __init__!")
        if self.backbone is None:
            raise RuntimeError("EfficientNetB3.backbone là None. Lỗi khởi tạo backbone!")
        
        try:
            return self.backbone(x)
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chạy forward pass trong backbone: {e}") from e


# --------- Factories (giữ tương thích) ----------
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
    # Test nhanh (không cần internet)
    m = EfficientNetB3(num_classes=2, drop_connect_rate=0.2, dropout=0.3, pretrained=False, freeze_backbone=True)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Model:", m.model_name, "| Output:", y.shape)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in m.parameters())
    print(f"Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")
