import math
import torch
from torch.nn import Conv2d, BatchNorm2d, PReLU, Sequential, Module

from models.encoders.helpers import get_blocks, bottleneck_IR, bottleneck_IR_SE
from models.stylegan2.model import EqualLinear


class WEncoder(Module):
    def __init__(self, num_layers, mode='ir', opts=None):
        super(WEncoder, self).__init__()
        print('Using WEncoder')
        assert num_layers in [50, 100, 152], 'num_layers should be 50,100, or 152'
        assert mode in ['ir', 'ir_se'], 'mode should be ir or ir_se'
        blocks = get_blocks(num_layers)
        if mode == 'ir':
            unit_module = bottleneck_IR
        elif mode == 'ir_se':
            unit_module = bottleneck_IR_SE
        self.input_layer = Sequential(Conv2d(3, 64, (3, 3), 1, 1, bias=False),
                                      BatchNorm2d(64),
                                      PReLU(64))
        self.output_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = EqualLinear(512, 512, lr_mul=1)
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(unit_module(bottleneck.in_channel,
                                           bottleneck.depth,
                                           bottleneck.stride))
        self.body = Sequential(*modules)
        log_size = int(math.log(opts.output_size, 2))
        self.style_count = 2 * log_size - 2

    def forward(self, x): #x[8,3,256,256]
        x = self.input_layer(x) #Conv-BN-ReLU [8,64,256,256]
        x = self.body(x) #[8,512,16,16]
        x = self.output_pool(x)#AdaptiveAvgPool [8,512,1,1]
        x = x.view(-1, 512)#[8,512]
        x = self.linear(x)#EqualLinear(512,512) [8,512]
        return x.repeat(self.style_count, 1, 1).permute(1, 0, 2) #重复18次 [8,18,512]，因为StyleGAN2中是18个style，但是这样看起来18个style是一样，看起来是在stylegan中再进行变换

