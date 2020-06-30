from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from attention_modules import CBAM, SqueezeExcitationBlock
from vam import VAM


class BasicConv(nn.Module):
    def __init__(self, input_ch, output_ch, kernel_size, padding=0, stride=1, use_batchnorm=True, groups=1,
                 attention=None, size=None):
        super(BasicConv, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(input_ch, output_ch, kernel_size, stride, padding,
                                            bias=False if use_batchnorm else True, groups=groups),
                                  nn.BatchNorm2d(output_ch),
                                  nn.ReLU(True))
        if attention == 'SE':
            self.conv.add_module("Attention", SqueezeExcitationBlock(output_ch))

        elif attention == "CBAM":
            self.conv.add_module("Attention", CBAM(output_ch))

        elif attention == 'VAM':
            self.conv.add_module("Attention", VAM(output_ch, size))


    def forward(self, x):
        return self.conv(x)


class MobileNet(nn.Module):
    def __init__(self, width_multiplier=1.0, input_ch=3, init_ch=32, dataset='CFIAR100', attention=None, group_size=2):
        super(MobileNet, self).__init__()
        if dataset in ['CIFAR10', 'SVHN']:
            n_classes = 10

        elif dataset == 'CIFAR100':
            n_classes = 100

        elif dataset == 'ImageNet1K':
            n_classes = 1000

        n_ch = int(init_ch * width_multiplier)
        conv = partial(BasicConv, attention=attention)
        self.network = nn.Sequential(conv(input_ch, n_ch, 3, padding=1, stride=2, size=112), # input size 224x224x3
                                     conv(n_ch, n_ch, 3, padding=1, groups=n_ch, size=112), # input size 112x112x32
                                     conv(n_ch, 2 * n_ch, 1, size=112), # input size 112x112x32
                                     conv(2 * n_ch, 2 * n_ch, 3, padding=1, stride=2, groups=2 * n_ch, size=56), # input size 112x112x64
                                     conv(2 * n_ch, 4 * n_ch, 1, size=56), # input size 56x56x64
                                     conv(4 * n_ch, 4 * n_ch, 3, padding=1, groups=4 * n_ch, size=56), # input size 56x56x128
                                     conv(4 * n_ch, 4 * n_ch, 1, size=56), # input size 56x56x128

                                     conv(4 * n_ch, 4 * n_ch, 3, padding=1, stride=2, groups=4 * n_ch, size=28), # input size 56x56x128
                                     conv(4 * n_ch, 8 * n_ch, 1, size=28),
                                     conv(8 * n_ch, 8 * n_ch, 3, padding=1, groups=8 * n_ch, size=28),
                                     conv(8 * n_ch, 8 * n_ch, 1, size=28),

                                     conv(8 * n_ch, 8 * n_ch, 3, padding=1, stride=2, groups=8 * n_ch, size=14),
                                     conv(8 * n_ch, 16 * n_ch, 1, size=14),

                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 1, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 1, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 1, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 1, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, groups=16 * n_ch, size=14),
                                     conv(16 * n_ch, 16 * n_ch, 1, size=14),

                                     conv(16 * n_ch, 16 * n_ch, 3, padding=1, stride=2, groups=16 * n_ch, size=7),
                                     conv(16 * n_ch, 32 * n_ch, 1, size=7),
                                     conv(32 * n_ch, 32 * n_ch, 3, padding=1, groups=32 * n_ch, size=7),
                                     conv(32 * n_ch, 32 * n_ch, 1, size=7),

                                     nn.AdaptiveAvgPool2d((1, 1)),
                                     View(-1),
                                     nn.Linear(32 * n_ch, n_classes))

        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class ResidualBlock(nn.Module):
    def __init__(self, input_ch, output_ch, bottle_neck_ch=0, pre_activation=False, first_conv_stride=1, n_groups=1,
                 attention='CBAM', group_size=2, size=56):
        super(ResidualBlock, self).__init__()
        act = nn.ReLU(inplace=True)
        norm = nn.BatchNorm2d

        if pre_activation:
            if bottle_neck_ch:
                block = [norm(input_ch),
                         act,
                         nn.Conv2d(input_ch, bottle_neck_ch, 1, bias=False)]  # Caffe version has stride 2 here

                block += [norm(bottle_neck_ch),
                          act,
                          nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, padding=1, stride=first_conv_stride,
                                    groups=n_groups, bias=False)]  # PyTorch version has stride 2 here

                block += [norm(bottle_neck_ch),
                          nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False)]

            else:
                block = [norm(input_ch),
                         act,
                         nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, groups=n_groups, padding=1,
                                   bias=False)]

                block += [norm(output_ch),
                          act,
                          nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False)]

        else:
            if bottle_neck_ch:
                block = [nn.Conv2d(input_ch, bottle_neck_ch, 1, bias=False),  # Caffe version has stride 2 here
                         norm(bottle_neck_ch),
                         act]
                block += [nn.Conv2d(bottle_neck_ch, bottle_neck_ch, 3, padding=1, stride=first_conv_stride,
                                    groups=n_groups,
                                    bias=False),  # PyTorch version has stride 2 here
                          norm(bottle_neck_ch),
                          act]
                block += [nn.Conv2d(bottle_neck_ch, output_ch, 1, bias=False),
                          norm(output_ch)]

            else:
                block = [nn.Conv2d(input_ch, output_ch, 3, stride=first_conv_stride, groups=n_groups, padding=1,
                                   bias=False),
                         norm(output_ch),
                         act]
                block += [nn.Conv2d(output_ch, output_ch, 3, padding=1, bias=False),
                          norm(output_ch)]

        if attention == 'CBAM':
            block += [CBAM(output_ch)]

        elif attention == 'SE':
            block += [SqueezeExcitationBlock(output_ch)]

        elif attention == 'VAM':
            block += [VAM(output_ch, size)]

        if input_ch != output_ch:
            side_block = [nn.Conv2d(input_ch, output_ch, 1, stride=first_conv_stride, bias=False),
                          norm(output_ch)]
            self.side_block = nn.Sequential(*side_block)
            self.varying_size = True

        else:
            self.varying_size = False

        self.block = nn.Sequential(*block)

    #     self.var = None
    #     # self.register_forward_hook(self.append_var)
    #
    # # def reset_list(self):
    # #     self.list_var.clear()
    #
    # def get_var(self):
    #     return self.var

    def forward(self, x):
        if self.varying_size:
            x = F.relu(self.side_block(x) + self.block(x))
            # self.var = torch.mean(torch.var(x, dim=1), dim=(1, 2))

            return x
        else:
            x = F.relu(x + self.block(x))
            # self.var = torch.mean(torch.var(x, dim=1), dim=(1, 2))

            return x


class ResidualNetwork(nn.Module):
    def __init__(self, n_layers=50, dataset='ImageNet1K', attention='TAM', group_size=2):
        super(ResidualNetwork, self).__init__()
        RB = partial(ResidualBlock, attention=attention, group_size=group_size)

        if dataset == 'ImageNet1K':
            network = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            n_classes = 1000
            init_size = 56

        elif dataset == 'CIFAR10':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 10
            init_size = 32

        elif dataset == 'CIFAR100':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 100
            init_size = 32

        elif dataset == 'SVHN':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 10
            init_size = 32

        else:
            """
            For other dataset
            """
            pass

        if n_layers == 18:
            network += [RB(64, 64, size=init_size),
                        RB(64, 64, size=init_size)]

            init_size //= 2

            network += [RB(64, 128, first_conv_stride=2, size=init_size),
                        RB(128, 128, size=init_size)]

            init_size //= 2
            network += [RB(128, 256, first_conv_stride=2, size=init_size),
                        RB(256, 256, size=init_size)]

            init_size //= 2
            network += [RB(256, 512, first_conv_stride=2, size=init_size),
                        RB(512, 512, size=init_size)]
            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 34:
            network += [RB(64, 64, size=init_size) for _ in range(3)]

            init_size //= 2

            network += [RB(64, 128, first_conv_stride=2, size=init_size)]
            network += [RB(128, 128, size=init_size) for _ in range(3)]

            init_size //= 2

            network += [RB(128, 256, first_conv_stride=2, size=init_size)]
            network += [RB(256, 256, size=init_size) for _ in range(5)]

            init_size //= 2

            network += [RB(256, 512, first_conv_stride=2, size=init_size)]
            network += [RB(512, 512, size=init_size) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(512, n_classes)]

        elif n_layers == 50:
            network += [RB(64, 256, bottle_neck_ch=64, size=init_size)]
            network += [RB(256, 256, bottle_neck_ch=64, size=init_size)]
            network += [RB(256, 256, bottle_neck_ch=64, size=init_size)]

            init_size //= 2

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2, size=init_size)]  # 28
            network += [RB(512, 512, bottle_neck_ch=128, size=init_size)]
            network += [RB(512, 512, bottle_neck_ch=128, size=init_size)]
            network += [RB(512, 512, bottle_neck_ch=128, size=init_size)]

            init_size //=2

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2, size=init_size)]  # 14
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size)]

            init_size //=2

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, size=init_size)]
            network += [RB(2048, 2048, bottle_neck_ch=512, size=init_size)]
            network += [RB(2048, 2048, bottle_neck_ch=512, size=init_size)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 101:
            network += [RB(64, 256, bottle_neck_ch=64, size=init_size)]
            network += [RB(256, 256, bottle_neck_ch=64, size=init_size) for _ in range(2)]

            init_size //= 2

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2, size=init_size)]
            network += [RB(512, 512, bottle_neck_ch=128, size=init_size) for _ in range(3)]


            init_size //=2

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256, size=init_size) for _ in range(22)]

            init_size //= 2

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, size=init_size)]
            network += [RB(2048, 2048, bottle_neck_ch=512, size=init_size) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        elif n_layers == 152:
            network += [RB(64, 256, bottle_neck_ch=64, size=init_size)]
            network += [RB(256, 256, bottle_neck_ch=64) for _ in range(2)]

            network += [RB(256, 512, bottle_neck_ch=128, first_conv_stride=2, size=init_size)]
            network += [RB(512, 512, bottle_neck_ch=128) for _ in range(7)]

            network += [RB(512, 1024, bottle_neck_ch=256, first_conv_stride=2, size=init_size)]
            network += [RB(1024, 1024, bottle_neck_ch=256) for _ in range(35)]

            network += [RB(1024, 2048, bottle_neck_ch=512, first_conv_stride=2, size=init_size)]
            network += [RB(2048, 2048, bottle_neck_ch=512) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)), View(-1), nn.Linear(2048, n_classes)]

        else:
            raise NotImplementedError("Invalid n_layers {}. Choose among 18, 34, 50, 101, or 152.".format(n_layers))

        self.network = nn.Sequential(*network)
        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    # def get_var_loss(self):
    #     loss = 0
    #     for m in self.network:
    #         try:
    #             loss += m.get_var()
    #
    #         except AttributeError:
    #             pass
    #     return torch.mean(1 / (loss + 1e-8))# 1 / (torch.mean(loss) + 1e-8) #-torch.log(loss.mean() + 1e-8)#

    def forward(self, x):
        return self.network(x)


class ResNext(nn.Module):
    def __init__(self, n_layers=50, n_groups=32, dataset='ImageNet', attention='SE', group_size=2):
        super(ResNext, self).__init__()
        RB = partial(ResidualBlock, attention=attention, group_size=group_size, n_groups=n_groups)
        if dataset == 'ImageNet1K':
            network = [nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            init_size = 56
            n_classes = 1000

        elif dataset == 'CIFAR10':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 10

        elif dataset == 'CIFAR100':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 100

        elif dataset == 'SVHN':
            network = [nn.Conv2d(3, 64, 3, padding=1, bias=False),
                       nn.BatchNorm2d(64),
                       nn.ReLU(inplace=True)]
            n_classes = 10

        if n_layers == 29:
            assert 'ImageNet' not in dataset
            network += [RB(64, 256, bottle_neck_ch=512)]
            network += [RB(256, 256, bottle_neck_ch=512) for _ in range(2)]

            network += [RB(256, 512, bottle_neck_ch=1024, first_conv_stride=2)]
            network += [RB(512, 512, bottle_neck_ch=1024) for _ in range(2)]

            network += [RB(512, 1024, bottle_neck_ch=2048, first_conv_stride=2)]
            network += [RB(1024, 1024, bottle_neck_ch=2048) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)),
                        View(-1),
                        nn.Linear(1024, n_classes)]

        elif n_layers == 50:
            network += [RB(64, 256, bottle_neck_ch=128, size=init_size)]
            network += [RB(256, 256, bottle_neck_ch=128, size=init_size) for _ in range(2)]

            init_size //= 2

            network += [RB(256, 512, bottle_neck_ch=256, first_conv_stride=2, size=init_size)]  # 28
            network += [RB(512, 512, bottle_neck_ch=256, size=init_size) for _ in range(3)]

            init_size //= 2

            network += [RB(512, 1024, bottle_neck_ch=512, first_conv_stride=2, size=init_size)]  # 14
            network += [RB(1024, 1024, bottle_neck_ch=512, size=init_size) for _ in range(5)]

            init_size //= 2

            network += [RB(1024, 2048, bottle_neck_ch=1024, first_conv_stride=2, size=init_size)]
            network += [RB(2048, 2048, bottle_neck_ch=1024, size=init_size) for _ in range(2)]

            network += [nn.AdaptiveAvgPool2d((1, 1)),
                        View(-1),
                        nn.Linear(2048, n_classes)]

        self.network = nn.Sequential(*network)
        self.apply(init_weights)

        print(self)
        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class WideResNet(nn.Module):
    def __init__(self, n_layers=16, widening_factor=8, dataset='ImageNet', attention='None', group_size=2):
        super(WideResNet, self).__init__()
        assert (n_layers - 4) % 6 == 0
        N = (n_layers - 4) // 6

        RB = partial(ResidualBlock, attention=attention, pre_activation=True, group_size=group_size)
        if dataset == 'ImageNet1K':
            network = [nn.Conv2d(3, 16, 7, stride=2, padding=3, bias=False),
                       nn.BatchNorm2d(16),
                       nn.ReLU(inplace=True),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
            init_ch = 16
            n_ch = int(init_ch * widening_factor)
            init_size = 56
            n_classes = 1000

        elif dataset == 'CIFAR10':
            network = [nn.Conv2d(3, 16, 3, padding=1, bias=False),
                       nn.BatchNorm2d(16),
                       nn.ReLU(inplace=True)]
            init_ch = 16
            n_ch = int(init_ch * widening_factor)
            n_classes = 10

        elif dataset == 'CIFAR100':
            network = [nn.Conv2d(3, 16, 3, padding=1, bias=False),
                       nn.BatchNorm2d(16),
                       nn.ReLU(inplace=True)]
            init_ch = 16
            n_ch = int(init_ch * widening_factor)
            n_classes = 100

        elif dataset == 'SVHN':
            network = [nn.Conv2d(3, 16, 3, padding=1, bias=False),
                       nn.BatchNorm2d(16),
                       nn.ReLU(inplace=True)]
            init_ch = 16
            n_ch = int(init_ch * widening_factor)
            n_classes = 10

        network += [RB(init_ch, n_ch, size=init_size)]
        for _ in range(N - 1):
            network += [RB(n_ch, n_ch, size=init_size)]

        init_size //= 2

        network += [RB(n_ch, 2 * n_ch, size=init_size, first_conv_stride=2)]
        for _ in range(N - 1):
            network += [RB(2 * n_ch, 2 * n_ch, size=init_size)]

        init_size //= 2

        network += [RB(2 * n_ch, 4 * n_ch, size=init_size, first_conv_stride=2)]
        for _ in range(N - 1):
            network += [RB(4 * n_ch, 4 * n_ch, size=init_size)]

        network += [nn.BatchNorm2d(4 * n_ch),
                    nn.ReLU(True)]

        network += [nn.AdaptiveAvgPool2d((1, 1)),
                    View(-1),
                    nn.Linear(4 * n_ch, n_classes)]

        self.network = nn.Sequential(*network)
        self.apply(init_weights)
        print(self)

        n_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print("The number of learnable parameters : {}".format(n_params))

    def forward(self, x):
        return self.network(x)


class View(nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.shape[0], *self.shape)


def init_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

    elif isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight)
