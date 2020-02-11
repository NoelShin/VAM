import torch
import torch.nn as nn


class VAM(nn.Module):
    def __init__(self, n_ch, group_size, size):
        super(VAM, self).__init__()
        init_ch = n_ch
        self.channel_branch = nn.Conv2d(n_ch, 1, 1, bias=False)
        self.spatial_branch = SpatialBranch(n_ch, size)
        
        self.bn_sp = nn.BatchNorm2d(n_ch)
        self.bn_ch = nn.BatchNorm2d(1)

    def forward(self, x):
        return x * torch.sigmoid(self.bn_sp(self.spatial_branch(x)) + self.bn_ch(self.channel_branch(x)))


class SpatialBranch(nn.Module):
    def __init__(self, n_ch, size):
        super(SpatialBranch, self).__init__()
        self.conv = nn.Conv1d(size * size, 1, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return x.view(n, c, 1, 1)