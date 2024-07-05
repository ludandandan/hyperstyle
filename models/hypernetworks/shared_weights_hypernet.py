import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class SharedWeightsHypernet(nn.Module):

    def __init__(self, f_size=3, z_dim=512, out_size=512, in_size=512, mode=None):#全都用默认的
        super(SharedWeightsHypernet, self).__init__()
        self.mode = mode #None
        self.z_dim = z_dim # 512
        self.f_size = f_size # 3，卷积核大小
        if self.mode == 'delta_per_channel':
            self.f_size = 1
        self.out_size = out_size # 512
        self.in_size = in_size # 512

        self.w1 = Parameter(torch.fmod(torch.randn((self.z_dim, self.out_size * self.f_size * self.f_size)).cuda() / 40, 2))#先取[512,512x3x3=4608]的正态分布的随机数，均值为0，方差为1，然后再除以40变小，然后取模2（就是对2取余数，那肯定是小于2），使结果在[-2,2]范围内
        self.b1 = Parameter(torch.fmod(torch.randn((self.out_size * self.f_size * self.f_size)).cuda() / 40, 2))#[4608]

        self.w2 = Parameter(torch.fmod(torch.randn((self.z_dim, self.in_size * self.z_dim)).cuda() / 40, 2))#[512,512x512=262144]
        self.b2 = Parameter(torch.fmod(torch.randn((self.in_size * self.z_dim)).cuda() / 40, 2)) #[262144]

    def forward(self, z): #通过输入一个潜在变量z，动态生成卷积核
        batch_size = z.shape[0] #z[8,512]
        h_in = torch.matmul(z, self.w2) + self.b2 #乘以权重w2再加上偏置b2，[8,512]x[512,262144]+[262144]=[8,262144]
        h_in = h_in.view(batch_size, self.in_size, self.z_dim)#[8,262144]->[8,512,512]

        h_final = torch.matmul(h_in, self.w1) + self.b1 #乘以权重w1再加上偏置b1，[8,512,512]x[512,4608]+[4608]=[8,512,4608]
        kernel = h_final.view(batch_size, self.out_size, self.in_size, self.f_size, self.f_size)#[8,512,4608]->[8,512,512,3,3]
        if self.mode == 'delta_per_channel':  # repeat per channel values to the 3x3 conv kernels
            kernel = kernel.repeat(1, 1, 1, 3, 3)
        return kernel  # [8,512,512,3,3]
