# xception.py  — pretrained Xception (timm)

import torch
import torch.nn as nn

class Xception(nn.Module):
    """
    Wrapper Xception dùng pretrained từ timm (ImageNet).
    - num_classes: số lớp output (vd 2 cho fake/real)
    - model_name: 'xception' (mặc định). Có thể dùng 'xception41','xception65','xception71','tf_xception'
    - pretrained: tải trọng số ImageNet sẵn
    - freeze_backbone: True -> chỉ train classifier vài epoch đầu (finetune nhanh & ổn định)
    - drop_rate: dropout ở head
    - drop_path_rate: stochastic depth
    Gợi ý input: 299x299 cho 'tf_xception' / 299 or 224 cho 'xception' (timm vẫn chạy với mọi size nhờ GAP).
    """
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

        # Tạo backbone từ timm, thay classifier để ra đúng num_classes
        # Một số model không hỗ trợ drop_path_rate, nên try-except
        try:
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                global_pool="avg"
            )
        except TypeError as e:
            # Fallback: không dùng drop_path_rate
            if "drop_path_rate" in str(e):
                print(f"Warning: {model_name} không hỗ trợ drop_path_rate, bỏ qua parameter này")
                self.backbone = timm.create_model(
                    model_name,
                    pretrained=pretrained,
                    num_classes=num_classes,
                    drop_rate=drop_rate,
                    global_pool="avg"
                )
            else:
                raise

        if freeze_backbone:
            # Đóng băng mọi thứ trừ classifier (tên head tuỳ model, timm map về "classifier")
            for n, p in self.backbone.named_parameters():
                if "classifier" in n:
                    continue
                p.requires_grad = False

        self._frozen = freeze_backbone
        self.model_name = model_name

    def unfreeze(self):
        """Gọi sau vài epoch để finetune toàn bộ backbone."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        self._frozen = False

    def forward(self, x):
        return self.backbone(x)


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
