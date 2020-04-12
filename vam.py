import torch
import torch.nn as nn


class VAM(nn.Module):
    def __init__(self, n_ch, group_size, size):
        super(VAM, self).__init__()
        init_ch = n_ch
        self.channel_branch = nn.Conv2d(n_ch, 1, 1, bias=False)
        self.spatial_branch = SpatialBranch(n_ch, size)
        
        self.ln_sp = nn.LayerNorm((n_ch, 1, 1)) # nn.BatchNorm2d(n_ch)
        self.ln_ch = nn.LayerNorm((1, size, size)) # nn.BatchNorm2d(1)
        # self.ln = nn.LayerNorm((n_ch, size, size), elementwise_affine=True)
    def forward(self, x):
        # return x * torch.sigmoid(self.ln(self.spatial_branch(x) + self.channel_branch(x)))
        return x * torch.sigmoid(self.ln_sp(self.spatial_branch(x)) + self.ln_ch(self.channel_branch(x)))
        # return x * torch.sigmoid(self.ln(self.spatial_branch(x) + self.channel_branch(x)))


class SpatialBranch(nn.Module):
    def __init__(self, n_ch, size):
        super(SpatialBranch, self).__init__()
        self.conv = nn.Conv1d(size * size, 1, kernel_size=1, bias=False)

    def forward(self, x):
        # print(x.shape)
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return x.view(n, c, 1, 1)
