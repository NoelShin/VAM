import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from PIL import Image


class TAM(nn.Module):
    def __init__(self, n_ch, group_size, size):
        super(TAM, self).__init__()
        # tam = [nn.Conv2d(n_ch, 1, 1, bias=False), nn.Sigmoid()]  #[nn.AdaptiveAvgPool2d((1, 1))]
        # tam = [nn.AdaptiveAvgPool2d(1)]
        # init_ch = n_ch
        # tam = [] # [nn.Conv2d(n_ch, n_ch, 3, padding=1, groups=n_ch), nn.ReLU(True)]
        # for i in range(int(log(n_ch) / log(group_size))):
        #     tam += [nn.Conv2d(n_ch, n_ch // group_size, 1, groups=n_ch // group_size, bias=True)]
        #     n_ch //= group_size
        #     tam += [nn.LeakyReLU(True)]
        #     if n_ch == group_size:
        #         break
        # tam += [nn.Conv2d(n_ch, init_ch, 1)]

        # if n_ch != 1:
        #     tam += [nn.Conv2d(n_ch, 1, 1, bias=False)]
        # tam += [nn.Sigmoid()]
        # self.channel_bank = ChannelBank()
        init_ch = n_ch
        # tam = []
        # tam += [nn.Conv2d(n_ch, 1, 1, bias=False),
        #         nn.InstanceNorm2d(1, affine=True),
        #         nn.Sigmoid()]
        # tam = [ChannelResidual(2), nn.BatchNorm2d(n_ch // 2), nn.Conv2d(n_ch // 2, 1, 1, bias=False)]
        self.channel_branch = nn.Conv2d(n_ch, 1, 1, bias=False)
        self.spatial_branch = SpatialBranch(n_ch, size)
        # self.bn = nn.BatchNorm2d(1)

#         for i in range(int(log(n_ch) / log(group_size))):
#             tam += [# EfficientTree(group_size),
#                     # nn.Conv2d(n_ch, n_ch // group_size, 1, groups=n_ch // group_size, bias=False),
#                     ChannelResidual(n_ch, group_size, size=size),
#                     nn.BatchNorm2d(n_ch // group_size),
#                     Print()]
#             n_ch //= group_size
#             if n_ch != group_size:
#                  tam += [nn.LeakyReLU(negative_slope=0.2, inplace=True)]
#
# #                 # tam += [nn.PReLU(num_parameters=n_ch, init=1.0)]#[nn.LeakyReLU(negative_slope=0.2, inplace=True)]
# #
#             if n_ch == group_size:
#                 break
# #
#         #tam += [nn.Conv2d(n_ch, init_ch, 1, bias=True)] #, nn.BatchNorm2d(init_ch)]
#         if n_ch != 1:
#             tam += [ChannelResidual(n_ch, group_size, size), Print()]
# #nn.Conv2d(n_ch, 1, 1, bias=False),
#
        # tam += [nn.Sigmoid()]



        # if n_ch != 1:
        #     tam += [nn.Conv2d(n_ch, 1, 1, groups=1, bias=False),
        #             nn.BatchNorm2d(1)]

        # tam += [nn.Sigmoid()]
        # tam = [nn.AdaptiveAvgPool2d(1)]
        # for i in range(4):
        #     tam += [nn.Conv2d(n_ch, n_ch // group_size, 1, groups=n_ch // group_size, bias=False)]
        #     n_ch //= group_size
        #     if i != 3:
        #         tam += [nn.ReLU(True)]
        #
        # # tam += [nn.Conv2d(n_ch, n_ch, 1, bias=False), nn.ReLU(True)]
        #
        # for i in range(4):
        #     tam += [nn.Conv2d(n_ch, n_ch * group_size, 1, groups=n_ch, bias=True)]
        #     if i != 3:
        #         tam += [nn.ReLU(True)]
        #     n_ch *= group_size
        # tam += [nn.Sigmoid()]

        # for i in range(4):#int(log(n_ch) / log(group_size))):
        #     # tam += [ChannelSummation(group_size)]
        #     tam += [nn.Conv2d(n_ch, n_ch // group_size, 1, groups=n_ch // group_size, bias=False)]
        #     n_ch //= group_size
        #
        #     if n_ch == 1:
        #         break
        #
        #     tam += [nn.ReLU(True)]  # [nn.PReLU(n_ch, init=1.0)]
        # if n_ch != 1:
        #     tam += [ChannelSummation(n_ch)]

        # tam += [nn.Conv2d(n_ch // 16, init_ch, 1, bias=False), nn.Sigmoid()]
        # tam += [nn.Sigmoid()]
        # self.tam = nn.Sequential(*tam)
        # self.avg = nn.AdaptiveAvgPool2d(1)
        # self.bn_sp = nn.BatchNorm2d(n_ch)
        # self.ins_norm = nn.InstanceNorm2d(1, affine=True)
        # self.ln = nn.LayerNorm([n_ch, 1, 1], elementwise_affine=True)
        self.bn_sp = nn.BatchNorm2d(n_ch, momentum=0.01)
        self.bn_ch = nn.BatchNorm2d(1, momentum=0.01)

    def forward(self, x):
        x = x * torch.sigmoid(self.bn_sp(self.spatial_branch(x)) + self.bn_ch(self.channel_branch(x)))
        return x # + self.bn_sp(self.spatial_branch(x)) + self.bn_ch(self.channel_branch(x))
        # y = self.ln(self.spatial_branch(x))
        # z = self.ins_norm(self.channel_branch(x * torch.sigmoid(y)))
        # z = self.ln(self.spatial_branch(x * torch.sigmoid(y)))

        # y = self.bn_sp(self.spatial_branch(x))
        # z = self.bn(self.channel_branch(x))
        # print("channel min: ", y.min(), "channel max: ", y.max())
        # print("spatial min: ", z.min(), "spatial max: ", z.max())
        # return x * torch.sigmoid(z)
        #x * torch.sigmoid(self.bn(self.channel_branch(x) + self.spatial_branch(x)))
        # _x = np.linspace(0, 55, 56)
        # y = np.linspace(0, 55, 56)
        # _x, y = np.meshgrid(_x, y)
        # z = self.tam(x)[0].squeeze().detach().cpu().numpy()
        #
        # fig = plt.figure()
        # ax = fig.gca(projection="3d")
        # ax.plot_surface(_x, y, z, cmap=cm.gray)
        # plt.show()
        # exit(100)
        # y = torch.mean(torch.var(x, dim=1, keepdim=True), dim=(2, 3), keepdim=True)
        # self.list_mean_var.append(y)
        # print(y.var(dim=(2, 3)))
        # n, _, _, _ = x.shape
        #
        # z = self.tam(x)
        # var_vec = torch.var(z, dim=(2, 3))
        # max_idx = torch.argmax(var_vec, dim=1)
        # y = torch.zeros_like(x)
        # for i in range(n):
        #     y[i, ...] = x[i, ...] * torch.sigmoid(z[i, max_idx[i], ...].unsqueeze(dim=0).unsqueeze(dim=0))
        # return x * torch.sigmoid(self.tam(x))

        # y = self.tam(x)
        # sample = y[1, ...].detach().cpu().numpy()
        # print(sample.min(), sample.max())
        # sample = 1 / (1 + np.exp(-sample))
        # plt.hist(sample.flatten())
        # plt.show()
        #
        # to_image(np.reshape(sample, (56, 56)), "TAM2")
        #
        # exit(10)

        # return y# * torch.sigmoid(y)
        # self.tam(x)
        # y = self.channel_bank.get_max_var_channel()
        # # print(y.shape)
        # self.channel_bank.reset()
        # return x * torch.sigmoid(y)  # x * torch.sigmoid(self.tam(x))
        # n, _, _, _ = x.shape
        # var_vec = torch.var(x, dim=(2, 3))
        # max_idx = torch.argmax(var_vec, dim=1)
        # min_idx = torch.argmin(var_vec, dim=1)
        # max_values = x[1, max_idx[1], ...]
        # max_values = max_values.detach().cpu().numpy()
        #
        # min_values = x[1, min_idx[1], ...]
        # min_values = min_values.detach().cpu().numpy()
        #
        # to_image(max_values, "abcd_max")
        # to_image(min_values, "abcd_min")
        #
        # # Image.fromarray()
        # exit(100)
        # y = torch.zeros_like(x)
        # for i in range(n):
        #     y[i, ...] = x[i, ...] * torch.sigmoid(x[i, idx[i], ...].unsqueeze(dim=0).unsqueeze(dim=0))
        #
        # return y


class ChannelResidual(nn.Module):
    def __init__(self, n_ch, group_size, size):
        super(ChannelResidual, self).__init__()
        self.group_size = group_size
        # self.conv = nn.Conv2d(n_ch, n_ch, 1, groups=n_ch, bias=False)
        self.conv = nn.Conv1d(size * size, size * size, 1, groups=size*size)

    def forward(self, x):
        n, c, h, w = x.shape
        # x = self.conv(x)
        x = x.view(n, c // self.group_size, self.group_size, h, w)
        x1, x2 = torch.split(x, 1, dim=2)  # n x c //2 x 1 x h x 2

        x1, x2 = x1.view(n, c // self.group_size, h * w), x2.view(n, c // self.group_size, h * w)  # n x c // 2 x hw
        x = self.conv(torch.min(x1, x2).transpose(1, 2))
        x = x.transpose(1, 2)

        return x.view(n, c // self.group_size, h, w)  # torch.min(x1, x2).squeeze(dim=2) #self.conv((x1 - x2).squeeze(dim=2))


class ChannelSummation(nn.Module):
    def __init__(self, n_ch, group_size):
        super(ChannelSummation, self).__init__()
        self.group_size = group_size
        self.conv = nn.Conv2d(n_ch // group_size, n_ch // group_size, 1, groups=n_ch // group_size, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        return self.conv(x.view(n, c // self.group_size, self.group_size, h, w).sum(2))


class EfficientTree(nn.Module):
    def __init__(self, group_size):
        super(EfficientTree, self).__init__()
        self.group_size = group_size
        self.conv = nn.Conv2d(group_size, 1, 1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, self.group_size, c // self.group_size, -1)  # n x g x c/g x hw
        x = self.conv(x)
        # x = self.conv(x.view(n, self.group_size, c // self.group_size, h * w))
        return x.squeeze().view(n, c // self.group_size, h, w)


class ChannelBank(nn.Module):
    def __init__(self):
        super(ChannelBank, self).__init__()
        self.list_channel = list()
        self.list_var = list()

    def forward(self, x):
        self.list_channel.append(x)
        # print(torch.var(x, dim=(2, 3)).detach().cpu().numpy().shape)
        self.list_var.append(torch.var(x, dim=(2, 3)).detach())
        return x

    def get_max_var_channel(self):
        idx = torch.cat(self.list_var, dim=1)
        x = torch.cat(self.list_channel, dim=1)
        idx_max = torch.argmax(idx, dim=1)
        idx_min = torch.argmin(idx, dim=1)
        # print(idx.shape, torch.argmax(idx, dim=1))
        # print(x.shape)
        # vals_max = x[0, torch.argmax(idx, dim=1)[0], :].flatten()
        # vals_max = vals_max.detach().cpu().numpy()
        # print("min: ", vals_max.min(), "max: ", vals_max.max())
        # plt.hist(vals_max, bins=np.arange(vals_max.min(), vals_max.max(), 0.05))
        # plt.show()
        # plt.close()
        # print(idx_max)
        # plt.hist(idx_max.detach().cpu().numpy(), bins=[0, 128, 192, 224, 240, 248, 252, 254, 255])
        # plt.show()
        # exit(100)
        #
        # vals_min = x[0, torch.argmin(idx, dim=1)[0], :].flatten()
        # vals_min = vals_min.detach().cpu().numpy()
        # print("min: ", vals_min.min(), "max: ", vals_min.max())
        # plt.hist(vals_min, bins=np.arange(vals_max.min(), vals_max.max(), 0.05))#, bins=np.arange(vals_max.min(), vals_max.max(), 0.05))
        # plt.show()
        #
        # channel_mean = x[0, :, :, :].mean(dim=0)
        # to_image(channel_mean.detach().cpu().numpy(), "channel_mean")
        # to_image(vals_max, "max")
        # to_image(vals_min, "min")
        #
        # exit(100)
        # x = self.list_channel[509]

        # print(np.array(self.list_var).shape)
        # print(np.argmax(self.list_var, axis=1))
        # print(torch.argmax(idx, dim=1).shape)
        idx = torch.argmax(idx, dim=1)  # B x 1
        return x[:, idx, ...].mean(dim=1, keepdim=True)
        # new_tensor = list()
        #
        # for i in range(idx.shape[0]):
        #     # print(x[i, idx[i], ...].shape)
        #     new_tensor.append(x[i, idx[i], :, :])
        # return torch.stack(new_tensor, dim=0).unsqueeze(1)
        # x = x[i, idx[i], :,  for i in range(len(idx))]
        # print(x.shape)
        # exit(100)
        # print(torch.index_select(x, dim=1, index=torch.argmax(idx, dim=1)).shape)
        # print(x[torch.argmax(idx, dim=1).unsqueeze(1)].shape)
        # return x[torch.argmax(idx, dim=1).unsqueeze(1), :, :]


def to_image(vals, name):
    vals -= vals.min()
    vals = vals / vals.max()
    vals *= 255.0
    vals = vals.astype(np.uint8)
    vals = vals.reshape((56, 56))
    Image.fromarray(vals).save("/userhome/shin_g/Desktop/{}.png".format(name))


class Print(nn.Module):
    def __init__(self):
        super(Print, self).__init__()

    def forward(self, x):
        # print("var: ", x[0, 0].var(dim=(0, 1)).squeeze().detach().item())
        # print("min: ", x[0, ...].min(), "max: ", x[0, ...].max())
        # print("var: ", x[0, ...].var(dim=(1, 2)), "mean: ", x[0].mean(dim=(1, 2)))

        return x


class SpatialBranch(nn.Module):
    def __init__(self, n_ch, size):
        super(SpatialBranch, self).__init__()
        self.conv = nn.Conv1d(size * size, 1, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        x = x.view(n, c, h * w).transpose(1, 2)
        x = self.conv(x).transpose(1, 2)
        return x.view(n, c, 1, 1)