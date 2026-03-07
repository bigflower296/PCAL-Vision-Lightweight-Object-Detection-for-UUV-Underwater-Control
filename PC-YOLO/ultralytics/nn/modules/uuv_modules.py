import torch
import torch.nn as nn
import torch.nn.functional as F
from .conv import Conv


# 1. UEM (Underwater Enhancement Module) - 放在最前端增强特征
class UEM(nn.Module):
    def __init__(self, c1, c2):
        super().__init__()
        # 确保输入输出通道一致，避免维度错误
        self.conv1 = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1, groups=c2)
        self.act = nn.SiLU()

    def forward(self, x):
        return x + self.act(self.conv2(self.conv1(x)))


# 2. PConv & C2f_Faster
class PConv(nn.Module):
    def __init__(self, dim, n_div=4, forward='split_cat', kernel_size=3):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, kernel_size, stride=1,
                                       padding=(kernel_size - 1) // 2, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x):
        x = x.clone()
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])
        return x

    def forward_split_cat(self, x):
        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        x1 = self.partial_conv3(x1)
        return torch.cat((x1, x2), 1)


class FasterBlock(nn.Module):
    def __init__(self, inc, dim, n_div=4, forward='split_cat'):
        super().__init__()
        self.dim = dim
        self.pconv = PConv(dim, n_div, forward)
        self.conv1 = nn.Conv2d(dim, dim, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(dim)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim, dim, 1, bias=False)
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.pconv(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = shortcut + self.drop_path(x)
        return x


class C2f_Faster(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(FasterBlock(self.c, self.c) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


# 3. RFB (Receptive Field Block) - 修复参数传递
class RFB(nn.Module):
    def __init__(self, in_planes, out_planes=None, stride=1, scale=0.1):
        super(RFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes or in_planes
        inter_planes = in_planes // 8

        # 使用 ultralytics 的 Conv，支持简写
        self.branch0 = nn.Sequential(
            Conv(in_planes, 2 * inter_planes, k=1),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=stride, p=1)
        )
        self.branch1 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1),
            Conv(inter_planes, 2 * inter_planes, k=3, s=1, p=1),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=stride, p=3, d=3)
        )
        self.branch2 = nn.Sequential(
            Conv(in_planes, inter_planes, k=1),
            Conv(inter_planes, (inter_planes // 2) * 3, k=3, s=1, p=1),
            Conv((inter_planes // 2) * 3, 2 * inter_planes, k=3, s=1, p=1),
            Conv(2 * inter_planes, 2 * inter_planes, k=3, s=stride, p=5, d=5)
        )
        self.conv_linear = Conv(6 * inter_planes, self.out_channels, k=1, s=1)
        self.shortcut = Conv(in_planes, self.out_channels, k=1, s=stride)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv_linear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        return F.relu(out)