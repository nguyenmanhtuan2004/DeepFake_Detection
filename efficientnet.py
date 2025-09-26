import math
import torch
import torch.nn as nn

# ---------- Tiện ích scale từ B0 -> B3 ----------
def round_filters(filters, width_mult, divisor=8):
    filters *= width_mult
    new_filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:
        new_filters += divisor
    return int(new_filters)

def round_repeats(repeats, depth_mult):
    return int(math.ceil(repeats * depth_mult))

# ---------- DropPath (Stochastic Depth) ----------
class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        mask = torch.empty(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
        return x / keep_prob * mask

# ---------- SE Block ----------
class SEBlock(nn.Module):
    def __init__(self, channels, se_ratio=0.25):
        super().__init__()
        reduced = max(1, int(channels * se_ratio))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=True),
            nn.SiLU(),
            nn.Conv2d(reduced, channels, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

# ---------- MBConv ----------
class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, expand_ratio, drop_prob=0.0, se_ratio=0.25):
        super().__init__()
        self.use_res = (s == 1 and in_ch == out_ch)
        mid = in_ch * expand_ratio

        layers = []
        # 1) Expand
        if expand_ratio != 1:
            layers += [
                nn.Conv2d(in_ch, mid, 1, bias=False),
                nn.BatchNorm2d(mid, eps=1e-3, momentum=0.01),
                nn.SiLU()
            ]
        # 2) Depthwise
        layers += [
            nn.Conv2d(mid, mid, k, stride=s, padding=k//2, groups=mid, bias=False),
            nn.BatchNorm2d(mid, eps=1e-3, momentum=0.01),
            nn.SiLU()
        ]
        self.pre_se = nn.Sequential(*layers)
        self.se = SEBlock(mid, se_ratio=se_ratio)

        # 3) Project
        self.project = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch, eps=1e-3, momentum=0.01),
        )
        self.drop_path = DropPath(drop_prob) if drop_prob > 0 else nn.Identity()

    def forward(self, x):
        identity = x
        x = self.pre_se(x)
        x = self.se(x)
        x = self.project(x)
        if self.use_res:
            x = self.drop_path(x) + identity
        return x

# ---------- EfficientNet-B3 (tối giản) ----------
class EfficientNetB3(nn.Module):
    """
    B3: width_mult=1.2, depth_mult=1.4, input ~300x300, dropout ~0.3
    Base (B0) block config: (t, c, n, s, k)
      (1,16,1,1,3), (6,24,2,2,3), (6,40,2,2,5),
      (6,80,3,2,3), (6,112,3,1,5), (6,192,4,2,5), (6,320,1,1,3)
    """
    def __init__(self, num_classes=1000, drop_connect_rate=0.2, dropout=0.3):
        super().__init__()
        width_mult = 1.2
        depth_mult = 1.4
        se_ratio = 0.25

        # B0 config
        blocks_args = [
            # t,  c,   n, s, k
            [1,  16,  1, 1, 3],
            [6,  24,  2, 2, 3],
            [6,  40,  2, 2, 5],
            [6,  80,  3, 2, 3],
            [6, 112,  3, 1, 5],
            [6, 192,  4, 2, 5],
            [6, 320,  1, 1, 3],
        ]

        # Scale channels & repeats cho B3
        for i, (t, c, n, s, k) in enumerate(blocks_args):
            c = round_filters(c, width_mult)
            n = round_repeats(n, depth_mult)
            blocks_args[i] = [t, c, n, s, k]

        # Stem
        stem_ch = round_filters(32, width_mult)   # -> 40 với B3
        self.stem = nn.Sequential(
            nn.Conv2d(3, stem_ch, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_ch, eps=1e-3, momentum=0.01),
            nn.SiLU()
        )

        # Blocks
        total_blocks = sum(n for _, _, n, _, _ in blocks_args)
        block_id = 0
        in_ch = stem_ch
        blocks = []
        for (t, c, n, s, k) in blocks_args:
            for i in range(n):
                stride = s if i == 0 else 1
                drop_prob = drop_connect_rate * block_id / total_blocks
                blocks.append(
                    MBConv(
                        in_ch, c, k, stride, expand_ratio=t,
                        drop_prob=drop_prob, se_ratio=se_ratio
                    )
                )
                in_ch = c
                block_id += 1
        self.blocks = nn.Sequential(*blocks)

        # Head
        head_ch = round_filters(1280, width_mult)  # -> 1536 với B3
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch, eps=1e-3, momentum=0.01),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(head_ch, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        return x

# --------- Quick test ----------
if __name__ == "__main__":
    model = EfficientNetB3(num_classes=2, drop_connect_rate=0.2, dropout=0.3)
    x = torch.randn(1, 3, 300, 300)  # B3 khuyến nghị ~300x300
    out = model(x)
    print("Params:", sum(p.numel() for p in model.parameters())/1e6, "M")
    print("Out shape:", out.shape)
