import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.extend([".", ".."])

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from options.test_options import TestOptions
from utils.common import tensor2im
from utils.domain_adaptation_utils import run_domain_adaptation
from utils.model_utils import load_model, load_generator


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'domain_adaptation_results') #保存跨域生成的图像的路径
    out_path_coupled = os.path.join(test_opts.exp_dir, 'domain_adaptation_coupled') #原图和生成图像两张图拼接的结果的路径

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts) #载入训练好的hperstyle模型（在ffhq上训练的）,里面存了state_dict、opts、latent_avg、global_step、best_val_loss

    restyle_e4e, restyle_opts = load_model(test_opts.restyle_checkpoint_path,#restyle_e4e_ffhq_encode.pt
                                           update_opts={"resize_outputs": test_opts.resize_outputs,
                                                        "n_iters_per_batch": test_opts.restyle_n_iterations},
                                           is_restyle_encoder=True)#载入模型参数 和 训练参数（在estyle_e4e_ffhq_encode.pt里面就存了state_dict、opts、latent_avg、discriminator_state_dict）
    finetuned_generator = load_generator(test_opts.finetuned_generator_checkpoint_path) #sketch_hq.pt

    print('Loading dataset for {}'.format(opts.dataset_type))
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    global_i = 0
    for input_batch in tqdm(dataloader): #input_batch[4,3,256,256]

        if global_i >= opts.n_images:#所有的图像个数
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()#input_cuda[4,3,256,256]
            result_batch, _ = run_domain_adaptation(input_cuda, net, opts, finetuned_generator, #net是hyperstyle模型，finetuned_generator是sketch_hq.pt估计是styglegan2在草图上训练得到的生成器
                                                    restyle_e4e, restyle_opts)#restyle_e4e起到了将输入图像转换为latent code的作用（那么为什么不用w_encoder作为生成器潜在编码的生成器了呢）

        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
        for i in range(input_batch.shape[0]):#对每一张图像

            im_path = dataset.paths[global_i]#图像的路径

            curr_result = tensor2im(result_batch[i]) #域适应生成的图像 1024x1024
            input_im = tensor2im(input_batch[i]) #输入的图像 256x256

            res_save_path = os.path.join(out_path_results, os.path.basename(im_path))#结果保存的路径
            curr_result.resize(resize_amount).save(res_save_path)#保存起来

            coupled_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))#成对的结果保存的路径
            res = np.concatenate([np.array(input_im.resize(resize_amount)), np.array(curr_result.resize(resize_amount))],
                                 axis=1) #原图和生成图像拼接在一起
            Image.fromarray(res).save(coupled_save_path) #保存起来
            global_i += 1


if __name__ == '__main__':
    run()
