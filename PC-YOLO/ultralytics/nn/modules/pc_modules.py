import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


# ============================================================
# 创新点 1: DPE (Differentiable Physics Encoder)
# ============================================================
class DPE(nn.Module):
    # 【修改点】增加 c2 参数，接住 YOLO 传来的输出通道数
    def __init__(self, c1, c2):
        super().__init__()
        # DPE 是残差结构，输入输出通道必须一致，所以我们主要用 c1
        # 这里的 c2 只是为了占位，防止报错

        # 可学习的物理参数
        self.beta = nn.Parameter(torch.rand(1, c1, 1, 1) * 0.1)
        self.A = nn.Parameter(torch.rand(1, c1, 1, 1))

        # 深度估计器
        self.depth_conv = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1),
            nn.Sigmoid()
        )
        self.conv_out = Conv(c1, c1, 1)

    def forward(self, x):
        # 1. 估计相对深度 d(x)
        d = self.depth_conv(x)

        # 2. 物理公式: 传输率 t(x) = e^(-beta * d(x))
        t = torch.exp(-self.beta * d)

        # 3. 物理公式: 恢复辐射 J(x) = (I(x) - A) / t + A
        J = (x - self.A) / (t + 1e-6) + self.A

        # 残差连接
        return self.conv_out(J + x)


# ============================================================
# 创新点 2: BLCA (Bi-Level Cognitive Attention)
# ============================================================
class BLCA(nn.Module):
    # 【修改点】同样增加 c2 参数，并把 ratio 放在最后
    def __init__(self, c1, c2, ratio=16):
        super().__init__()
        # Level 1: 显著性流
        self.saliency = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Conv2d(c1, c1 // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(c1 // ratio, c1, 1, bias=False),
            nn.Sigmoid()
        )

        # Level 2: 任务流
        self.task = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 1, 1, groups=c1),
            nn.SiLU(),
            nn.Conv2d(c1, c1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        w_sal = self.saliency(x)
        w_task = self.task(x)
        return x * w_sal * w_task