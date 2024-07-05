from torch import nn
from torch.nn import BatchNorm2d, PReLU, Sequential, Module
from torchvision.models import resnet34

from models.hypernetworks.refinement_blocks import HyperRefinementBlock, RefinementBlock, RefinementBlockSeparable
from models.hypernetworks.shared_weights_hypernet import SharedWeightsHypernet


class SharedWeightsHyperNetResNet(Module):

    def __init__(self, opts):
        super(SharedWeightsHyperNetResNet, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(opts.layers_to_tune) == 0: #opts.layers_to_tune，'0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24'，就是stylegan2的1+16个conv层，就是不包括to_rgb层
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))#self.layers_to_tune是一个list，包含了上面的17个层[0,2,3,5,6,8,9,11,12,14,15,17,18,20,21,23,24]
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12] #共享层，非to_rgb层的前9层
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)#里面定义了两个全连接层的权重[512,512x3x3=4608]、[512,512x512=262144]和偏置[4608]、[262144]

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs # 26
        for layer_idx in range(self.n_outputs): #对26个层进行遍历
            if layer_idx in self.layers_to_tune:#如果在layers_to_tune中，也就是需要微调的层，也就是非to_rgb层，共17层
                if layer_idx in self.shared_layers:#如果是共享层，就用HyperRefinementBlock
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
                else:#否则就用RefinementBlock
                    refinement_block = RefinementBlock(layer_idx, opts, n_channels=512, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x): # 根据源图像+重建图像的特征，来计算微调stylegan2的参数的增量
        x = self.conv1(x) #这是普通的nn.Conv [8,6,256,256]->[8,64,128,128]
        x = self.bn1(x)#普通的BN[8,64,128,128]->[8,64,128,128]
        x = self.relu(x)#普通的PReLU[8,64,128,128]->[8,64,128,128]
        x = self.body(x)#self.body是resnet34的前四层，包括0-15个BasicBlock [8,64,128,128]->[8,512,16,16]
        weight_deltas = []
        for j in range(self.n_outputs): #self.n_outputs=26，就是stylegan2的26层（1conv+1to_rgb+16conv+8to_rgb）
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)#所有block的输入x都一样，都来自源图像+重建图像，都是[8,512,16,16]
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas #是一个list，共26个元素，0号元素torch.Size([8, 512, 512, 3, 3])，1号元素是None，就是stylegan2对应层的参数的增量（对于to_rgb层不计算）

class SharedWeightsHyperNetResNetSeparable(Module): #这个默认不用

    def __init__(self, opts):
        super(SharedWeightsHyperNetResNetSeparable, self).__init__()

        self.conv1 = nn.Conv2d(opts.input_nc, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = BatchNorm2d(64)
        self.relu = PReLU(64)

        resnet_basenet = resnet34(pretrained=True)
        blocks = [
            resnet_basenet.layer1,
            resnet_basenet.layer2,
            resnet_basenet.layer3,
            resnet_basenet.layer4
        ]
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(bottleneck)
        self.body = Sequential(*modules)

        if len(opts.layers_to_tune) == 0:
            self.layers_to_tune = list(range(opts.n_hypernet_outputs))
        else:
            self.layers_to_tune = [int(l) for l in opts.layers_to_tune.split(',')]

        self.shared_layers = [0, 2, 3, 5, 6, 8, 9, 11, 12]
        self.shared_weight_hypernet = SharedWeightsHypernet(in_size=512, out_size=512, mode=None)

        self.refinement_blocks = nn.ModuleList()
        self.n_outputs = opts.n_hypernet_outputs
        for layer_idx in range(self.n_outputs):
            if layer_idx in self.layers_to_tune:
                if layer_idx in self.shared_layers:#如果是共享层，就用HyperRefinementBlock
                    refinement_block = HyperRefinementBlock(self.shared_weight_hypernet, n_channels=512, inner_c=128)
                else: #否则就用RefinementBlockSeparable
                    refinement_block = RefinementBlockSeparable(layer_idx, opts, n_channels=512, inner_c=256)
            else:
                refinement_block = None
            self.refinement_blocks.append(refinement_block)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.body(x)
        weight_deltas = []
        for j in range(self.n_outputs):
            if self.refinement_blocks[j] is not None:
                delta = self.refinement_blocks[j](x)
            else:
                delta = None
            weight_deltas.append(delta)
        return weight_deltas
