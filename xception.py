import torch
import torch.nn as nn

# -------------------- Tiện ích: Separable Conv (Depthwise + Pointwise) --------------------
class SeparableConv2d(nn.Module):
    """
    Depthwise separable conv như trong Xception:
    - Depthwise: groups = in_channels
    - Pointwise: 1x1 để trộn kênh
    Thứ tự theo paper: ReLU -> SeparableConv -> BN (đặt ReLU ngoài block).
    Ở đây mình để conv + BN, còn ReLU được đặt trước khi gọi block (pre-activation style trong các flow).
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=bias)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return x

# -------------------- Xception Block (một residual unit) --------------------
class XceptionBlock(nn.Module):
    """
    Một block chuẩn gồm:
    - (ReLU -> SepConv -> ReLU -> SepConv -> [ReLU -> SepConv])  tùy số layers
    - Optional: downsample bằng stride ở conv cuối (khớp paper: stride 2 ở một số block)
    - Skip: Identity hoặc 1x1 Conv (stride theo block) để match shape
    """
    def __init__(self, in_ch, out_ch, reps, stride=1, grow_first=True):
        super().__init__()
        assert reps >= 1
        layers = []
        ch_in = in_ch
        ch_out = out_ch

        # “Grow first”: conv đầu tiên tăng số kênh lên out_ch, theo paper
        if grow_first:
            layers += [
                nn.ReLU(inplace=True),
                SeparableConv2d(ch_in, ch_out, 3, 1, 1),
            ]
            ch_in = ch_out

        # Các repetition ở giữa (nếu có)
        for _ in range(reps - 1):
            layers += [
                nn.ReLU(inplace=True),
                SeparableConv2d(ch_in, ch_in, 3, 1, 1),
            ]

        # Conv cuối cùng: có thể downsample bằng stride (áp dụng ở một số block entry/exit)
        layers += [
            nn.ReLU(inplace=True),
            SeparableConv2d(ch_in, ch_out, 3, stride=stride, padding=1),
        ]

        self.body = nn.Sequential(*layers)

        # Skip connection
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        return self.body(x) + self.skip(x)

# -------------------- Xception nguyên bản (ImageNet) --------------------
class Xception(nn.Module):
    """
    Cấu trúc theo paper (phiên bản ImageNet):
    Entry Flow:
      - Conv 3x3 s2, 32; Conv 3x3, 64
      - Block: (in=64,out=128,reps=2,stride=2)
      - Block: (in=128,out=256,reps=2,stride=2)
      - Block: (in=256,out=728,reps=2,stride=2)
    Middle Flow:
      - 8 blocks giống nhau: (in=728,out=728,reps=3,stride=1)
    Exit Flow:
      - Block: (in=728,out=1024,reps=2,stride=2)
      - ReLU -> SepConv(1024->1536) -> ReLU -> SepConv(1536->2048)
    Cuối: GlobalAvgPool -> FC(num_classes)
    Input chuẩn: 299x299
    """
    def __init__(self, num_classes=1000):
        super().__init__()

        # ----- Entry flow -----
        self.entry_conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=0, bias=False),  # 299->149
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.entry_conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0, bias=False),  # 149->147
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Ba block với stride=2 theo paper
        self.block1 = XceptionBlock(64,   128, reps=2, stride=2, grow_first=True)   # 147->74
        self.block2 = XceptionBlock(128,  256, reps=2, stride=2, grow_first=True)   # 74->37
        self.block3 = XceptionBlock(256,  728, reps=2, stride=2, grow_first=True)   # 37->19

        # ----- Middle flow (8 lần) -----
        middle_blocks = []
        for _ in range(8):
            middle_blocks.append(XceptionBlock(728, 728, reps=3, stride=1, grow_first=True))
        self.middle = nn.Sequential(*middle_blocks)

        # ----- Exit flow -----
        self.block_exit = XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False)  # 19->10

        self.exit_conv = nn.Sequential(
            nn.ReLU(inplace=True),
            SeparableConv2d(1024, 1536, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048, kernel_size=3, stride=1, padding=1),
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d,)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.entry_conv1(x)
        x = self.entry_conv2(x)

        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.middle(x)

        x = self.block_exit(x)
        x = self.exit_conv(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# -------------------- Quick test --------------------
if __name__ == "__main__":
    model = Xception(num_classes=2)
    x = torch.randn(1, 3, 299, 299)  # kích thước chuẩn của paper
    y = model(x)
    print("Output:", y.shape)
    print("Params (M):", sum(p.numel() for p in model.parameters()) / 1e6)
