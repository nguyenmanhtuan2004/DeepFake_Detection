# xception.py  — pretrained Xception (timm)

import torch
import torch.nn as nn

class Xception(nn.Module):
    def __init__(self,
                 num_classes: int = 2,
                 model_name: str = "xception",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
        super().__init__()
        try:
            import timm
        except ImportError as e:
            raise ImportError("Cần cài `timm`: pip install timm") from e

        # Kiểm tra model có tồn tại không
        available_models = timm.list_models()
        if model_name not in available_models:
            # Tìm model Xception thay thế
            xception_models = [m for m in available_models if 'xception' in m.lower()]
            if xception_models:
                print(f"[Xception] Model '{model_name}' không tồn tại. Các model Xception có sẵn: {xception_models}")
                model_name = xception_models[0]  # Dùng model đầu tiên
                print(f"[Xception] Sử dụng '{model_name}' thay thế")
            else:
                raise ValueError(f"Không tìm thấy model Xception nào trong timm. Available models: {len(available_models)}")

        # Khởi tạo backbone với debug và fallback
        self.backbone = None
        
        try:
            print(f"[Xception] Đang tạo model '{model_name}' với pretrained={pretrained}")
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                global_pool="avg"
            )
            print(f"[Xception] ✅ Tạo thành công!")
        except TypeError as e:
            print(f"[Xception] ❌ Lỗi TypeError: {e}")
            # Fallback: không dùng drop_path_rate
            if "drop_path_rate" in str(e):
                print(f"[Xception] 🔄 Thử bỏ drop_path_rate...")
                try:
                    self.backbone = timm.create_model(
                        model_name,
                        pretrained=pretrained,
                        num_classes=num_classes,
                        drop_rate=drop_rate,
                        global_pool="avg"
                    )
                    print(f"[Xception] ✅ Fallback thành công!")
                except Exception as ex2:
                    print(f"[Xception] ❌ Fallback thất bại: {ex2}")
                    raise RuntimeError(f"Không thể tạo Xception với '{model_name}'. "
                                     f"Lỗi gốc: {e}. Lỗi fallback: {ex2}") from ex2
            else:
                raise RuntimeError(f"Không thể tạo Xception với '{model_name}': {e}") from e
        except Exception as ex:
            print(f"[Xception] ❌ Lỗi khác: {ex}")
            if pretrained:
                print("[Xception] 🔄 Thử fallback với pretrained=False...")
                try:
                    self.backbone = timm.create_model(
                        model_name,
                        pretrained=False,
                        num_classes=num_classes,
                        drop_rate=drop_rate,
                        global_pool="avg"
                    )
                    print(f"[Xception] ✅ Fallback thành công!")
                except Exception as ex2:
                    print(f"[Xception] ❌ Fallback cũng thất bại: {ex2}")
                    raise RuntimeError(f"Không thể tạo Xception với '{model_name}'. "
                                     f"Lỗi gốc: {ex}. Lỗi fallback: {ex2}") from ex2
            else:
                raise RuntimeError(f"Không thể tạo Xception với '{model_name}': {ex}") from ex
        
        # Đảm bảo backbone đã được tạo
        if self.backbone is None:
            raise RuntimeError("Xception backbone vẫn là None sau khi khởi tạo!")

        if freeze_backbone:
            # Đóng băng mọi thứ trừ classifier (tên head tuỳ model, timm map về "classifier")
            for n, p in self.backbone.named_parameters():
                if "classifier" in n:
                    continue
                p.requires_grad = False

        self._frozen = freeze_backbone
        self.model_name = model_name
        
        # Validation để đảm bảo model sẵn sàng
        self._validate_model()

    def _validate_model(self):
        """Validate model is properly initialized"""
        if not hasattr(self, 'backbone') or self.backbone is None:
            raise RuntimeError("Xception backbone validation failed!")
        
        # Test forward pass với dummy input
        try:
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224)
                _ = self.backbone(dummy_input)
            print("[Xception] ✅ Model validation passed")
        except Exception as e:
            raise RuntimeError(f"Xception validation failed: {e}") from e

    def unfreeze(self):
        """Gọi sau vài epoch để finetune toàn bộ backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x):
        # DataParallel-safe forward: trực tiếp gọi backbone thay vì check hasattr
        # hasattr có thể fail khi model được replicate sang GPU khác
        try:
            return self.backbone(x)
        except AttributeError:
            raise RuntimeError(
                "Xception.backbone không tồn tại. Có thể do:\n"
                "1. Lỗi trong __init__ (backbone chưa được tạo)\n"
                "2. DataParallel replication issues\n"
                "3. Model được load không đúng cách"
            )
        except Exception as e:
            raise RuntimeError(f"Lỗi khi chạy forward pass trong Xception backbone: {e}") from e


# Factory để tương thích với trainer gọi create_model(...)
def create_model(num_classes=2,
                 model_name: str = "xception",
                 pretrained: bool = True,
                 freeze_backbone: bool = False,
                 drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0):
    return Xception(num_classes=num_classes,
                    model_name=model_name,
                    pretrained=pretrained,
                    freeze_backbone=freeze_backbone,
                    drop_rate=drop_rate,
                    drop_path_rate=drop_path_rate)


# -------------------- Quick test --------------------
if __name__ == "__main__":
    # Chạy nhanh với 224 (vẫn OK do GAP)
    m = Xception(num_classes=2, model_name="xception", pretrained=False)
    x = torch.randn(1, 3, 224, 224)
    y = m(x)
    print("Output:", y.shape)
    print("Params (M):", sum(p.numel() for p in m.parameters()) / 1e6)

    # Nếu muốn đúng “TF Xception” 299x299:
    m_tf = Xception(num_classes=2, model_name="tf_xception", pretrained=False)
    y_tf = m_tf(torch.randn(1, 3, 299, 299))
    print("TF-Xception Output:", y_tf.shape)
