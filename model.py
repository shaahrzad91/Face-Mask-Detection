import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU_block(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# Squeeze-and-Excitation Networks
# https://arxiv.org/abs/1709.01507
class SE_block(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)

    def forward(self, x):
        bs = x.size(0)
        out = self.avg_pool(x).view(bs, -1)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = torch.sigmoid(out)
        out = out.view(bs, -1, 1, 1)
        return x * out


class MaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.mods = nn.Sequential(
            ConvBNReLU_block(3, 128, 3, stride=1, padding=1, dilation=1, bias=True),
            ConvBNReLU_block(128, 128, 3, stride=1, padding=1, dilation=1, bias=True),
            SE_block(128),
            ConvBNReLU_block(128, 128, 3, stride=2, padding=1, dilation=1, bias=True),
            ConvBNReLU_block(128, 64, 3, stride=1, padding=1, dilation=2, bias=True),
            SE_block(64),
            ConvBNReLU_block(64, 64, 3, stride=1, padding=1, dilation=1, bias=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(64, 3),
        )

    def forward(self, x):
        return self.mods(x)
