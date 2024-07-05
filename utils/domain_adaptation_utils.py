import sys
sys.path.extend(['.', '..'])

from utils.inference_utils import run_inversion
from utils import restyle_inference_utils


def run_domain_adaptation(inputs, net, opts, fine_tuned_generator, restyle_e4e, restyle_opts):
    """ Combine restyle e4e's latent code with HyperStyle's predicted weight offsets. """ #将restyle e4e的潜在代码与HyperStyle的预测权重偏移组合在一起
    y_hat, latents = restyle_inference_utils.run_on_batch(inputs, restyle_e4e, restyle_opts)#inputs[4,3,256,256],y_hat[4,3,256,256],latents[4,18,512]，这里是将输入图像通过restyle e4e的编码器和解码器得到潜在编码和重建图像（这里会迭代2次）
    y_hat2, _, weights_deltas, _ = run_inversion(inputs, net, opts)#net是hyperstyle模型，将输入给到hyperstyle模型，然后得到模型预测的对当前输入stylegan2模型各层参数的增量，为什么生成的weights_deltas里面前几层都是None呢？因为在载入的hyperstyle_ffhq.pt中的参数--layers_to_tune, default='5,6,8,9,11,12,14,15,17,18,20,21,23,24'#返回两次微调stylegan2的生成器得到的重建图像，潜在编码和两次微调累积的权重增量
    weights_deltas = filter_non_ffhq_layers_in_toonify_model(weights_deltas)#把前面几层设置为None
    return fine_tuned_generator([latents], #finetuned_generator是sketch_hq.pt估计是styglegan2在草图上训练得到的生成器
                                input_is_latent=True,#将上面针对这些图像得到的weights_deltas（这个是2次迭代累积的）应用到针对sketch的stylegan2生成器上得到的重建图像
                                randomize_noise=True,
                                return_latents=True,
                                weights_deltas=weights_deltas)#那么就是把在ffhq照片训练的stylegan2的生成器的weights_deltas应用在在草图上训练的styglegan2生成器上，目前是只应用8层14, 15, 17, 18, 20, 21, 23, 24


def filter_non_ffhq_layers_in_toonify_model(weights_deltas):
    toonify_ffhq_layer_idx = [14, 15, 17, 18, 20, 21, 23, 24]  # convs 8-15 according to model_utils.py
    for i in range(len(weights_deltas)):
        if weights_deltas[i] is not None and i not in toonify_ffhq_layer_idx:
            weights_deltas[i] = None
    return weights_deltas

