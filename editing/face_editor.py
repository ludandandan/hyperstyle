import torch

from configs.paths_config import edit_paths
from utils.common import tensor2im


class FaceEditor:

    def __init__(self, stylegan_generator):
        self.generator = stylegan_generator
        self.interfacegan_directions = {
            'age': torch.load(edit_paths['age']).cuda(),
            'smile': torch.load(edit_paths['smile']).cuda(),
            'pose': torch.load(edit_paths['pose']).cuda()
        }

    def apply_interfacegan(self, latents, weights_deltas, direction, factor=1, factor_range=None):
        edit_latents = []
        direction = self.interfacegan_directions[direction]#[1,512]，指示age或pose或simle变化时，向量中各个元素的变化方向和程度
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            for f in range(*factor_range):#-5,-4,-3,-2,-1,0,1,2,3,4
                edit_latent = latents + f * direction#latents[4,18,512],direction[1,512]
                edit_latents.append(edit_latent)#结束之后edit_latents是含有10个元素的列表，每个元素[4,18,512]
            edit_latents = torch.stack(edit_latents).transpose(0,1)#[4,10,18,512]编辑后的潜在代码，每个图像编辑了10种程度
        else:
            edit_latents = latents + factor * direction
        return self._latents_to_image(edit_latents, weights_deltas)#给编辑后的潜在代码和权重增量，返回编辑后的图像

    def _latents_to_image(self, all_latents, weights_deltas):
        sample_results = {}
        with torch.no_grad():
            for idx, sample_latents in enumerate(all_latents):#[10,18,512]
                sample_deltas = [d[idx] if d is not None else None for d in weights_deltas]#取出当前图像对应的权重增量，weights_deltas是26个元素的list，每个元素是[4,512,512,3,3]类似，sample_deltas是26个元素的list，每个元素是[512,512,3,3]类似
                images, _ = self.generator([sample_latents],#当前图像经过10种程度编辑后的潜在代码[10,18,512]
                                           weights_deltas=sample_deltas, #当前10张图像对应的权重增量
                                           randomize_noise=False,
                                           input_is_latent=True)#输出微调后的stylegan2根据潜在代码生成的图像[10,3,1024,1024]
                sample_results[idx] = [tensor2im(image) for image in images]#把10张tensor图像转换为PIL图像
        return sample_results
