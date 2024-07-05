import numpy as np
from torch import nn
from torch.nn import Conv2d, Sequential, Module

from models.encoders.helpers import SeparableBlock
from models.stylegan2.model import EqualLinear


# layer_idx: [kernel_size, in_channels, out_channels]
PARAMETERS = {
    0: [3, 512, 512],
    1: [1, 512, 3],
    2: [3, 512, 512],
    3: [3, 512, 512],
    4: [1, 512, 3],
    5: [3, 512, 512],
    6: [3, 512, 512],
    7: [1, 512, 3],
    8: [3, 512, 512],
    9: [3, 512, 512],
    10: [1, 512, 3],
    11: [3, 512, 512],
    12: [3, 512, 512],
    13: [1, 512, 3],
    14: [3, 512, 256],
    15: [3, 256, 256],
    16: [1, 256, 3],
    17: [3, 256, 128],
    18: [3, 128, 128],
    19: [1, 128, 3],
    20: [3, 128, 64],
    21: [3, 64, 64],
    22: [1, 64, 3],
    23: [3, 64, 32],
    24: [3, 32, 32],
    25: [1, 32, 3]
}
TO_RGB_LAYERS = [1, 4, 7, 10, 13, 16, 19, 22, 25]


class RefinementBlock(Module):

    def __init__(self, layer_idx, opts, n_channels=512, inner_c=256, spatial=16):
        super(RefinementBlock, self).__init__()
        self.layer_idx = layer_idx #14,15,17,18,20,21,23,24一个一个来
        self.opts = opts
        self.kernel_size, self.in_channels, self.out_channels = PARAMETERS[self.layer_idx]#从字典里取，14层是3，512，256，这个字典记录了stylegan2的每一层的卷积核大小、输入通道数、输出通道数
        self.spatial = spatial #默认是16
        self.n_channels = n_channels #512
        self.inner_c = inner_c #256
        self.out_c = 512
        num_pools = int(np.log2(self.spatial)) - 1 #默认算出来是3
        if self.kernel_size == 3:
            num_pools = num_pools - 1 #又变成2
        self.modules = []
        self.modules += [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]#512->256，下采样
        for i in range(num_pools - 1):
            self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]#256->256，下采样
        self.modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()] #256->512，下采样
        self.convs = nn.Sequential(*self.modules)

        if layer_idx in TO_RGB_LAYERS:#如果现在来的是to_rgb层，但是一般不会来
            self.output = Sequential(
                Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1, stride=1, padding=0))
        else: #一般走这里
            self.output = Sequential(nn.AdaptiveAvgPool2d((1, 1)), #自适应平均池化
                                     Conv2d(self.out_c, self.in_channels * self.out_channels, kernel_size=1, stride=1,
                                            padding=0)) #512->512x256，是1x1卷积

    def forward(self, x): #[8,512,16,16]
        x = self.convs(x) #[8,512,16,16]->[8,512,2,2]
        x = self.output(x)#[8,512,2,2]->[8,131072,1,1] 131072=512x256
        if self.layer_idx in TO_RGB_LAYERS: #如果在to_rgb层，一般不走这里，to_rgb层对应的是None，是不进行参数微调的
            x = x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        else:
            x = x.view(-1, self.out_channels, self.in_channels)#[8,131072]->[8,256,512]
            x = x.unsqueeze(3).repeat(1, 1, 1, self.kernel_size).unsqueeze(4).repeat(1, 1, 1, 1, self.kernel_size)
        return x#[8,256,512]->[8,256,512,1]->[8,256,512,3]->[8,256,512,3,1]->[8,256,512,3,3],就是把x经过一系列卷积之后的结果，扩展成为一个卷积核（适应stylegan2对应层（主要是非to_rgb的后8层）的卷积核大小）


class HyperRefinementBlock(Module):
    def __init__(self, hypernet, n_channels=512, inner_c=128, spatial=16):
        super(HyperRefinementBlock, self).__init__()
        self.n_channels = n_channels#512
        self.inner_c = inner_c #128
        self.out_c = 512
        num_pools = int(np.log2(spatial))#4
        modules = [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=1, padding=1), nn.LeakyReLU()]#conv的通道512->128，不下采样
        for i in range(num_pools - 1):
            modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]#经过3个下采样的conv+lrelu，通道128不变
        modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]#经过一个下采样的conv+lrelu，通道128->512
        self.convs = nn.Sequential(*modules) #一共5个conv+lrelu，通道512->512，下采样4次
        self.linear = EqualLinear(self.out_c, self.out_c, lr_mul=1)#  [512,512]
        self.hypernet = hypernet

    def forward(self, features):
        code = self.convs(features) # [8, 512, 16, 16] 经过5个conv+lrelu ->[8, 512, 1, 1]
        code = code.view(-1, self.out_c)#self.out_c=512, [8, 512]
        code = self.linear(code)#再经过一个512-512的全连接层 [8, 512]
        weight_delta = self.hypernet(code) #[8, 512, 512, 3, 3]将code潜在变量输入到SharedWeightsHypernet（实际上就是两个全连接的权重和偏置）中，得到卷积核（有点奇怪，为什么卷积核还有8batch呢）
        return weight_delta # [8, 512, 512, 3, 3]，就是将输入features经过一系列的卷积和全连接操作（这里的全连接操作在非to_rgb层的前9层是共享权重的，通过只是实例化SharedWeightsHypernet做到共享）之后得到的结果，变成stylegan2对应层的卷积核大小，用于对stylegan2对应层的参数进行微调


class RefinementBlockSeparable(Module):

    def __init__(self, layer_idx, opts, n_channels=512, inner_c=256, spatial=16):
        super(RefinementBlockSeparable, self).__init__()
        self.layer_idx = layer_idx
        self.kernel_size, self.in_channels, self.out_channels = PARAMETERS[self.layer_idx]
        self.spatial = spatial
        self.n_channels = n_channels
        self.inner_c = inner_c
        self.out_c = 512
        num_pools = int(np.log2(self.spatial)) - 1
        self.modules = []
        self.modules += [Conv2d(self.n_channels, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for i in range(num_pools - 1):
            self.modules += [Conv2d(self.inner_c, self.inner_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.modules += [Conv2d(self.inner_c, self.out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*self.modules)

        self.opts = opts
        if self.layer_idx in TO_RGB_LAYERS:
            self.output = Sequential(Conv2d(self.out_c, self.in_channels * self.out_channels,
                                            kernel_size=1, stride=1, padding=0))
        else:
            self.output = Sequential(SeparableBlock(input_size=self.out_c,
                                                    kernel_channels_in=self.in_channels,
                                                    kernel_channels_out=self.out_channels,
                                                    kernel_size=self.kernel_size))

    def forward(self, x):
        x = self.convs(x)
        x = self.output(x)
        if self.layer_idx in TO_RGB_LAYERS:
            x = x.view(-1, self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        return x