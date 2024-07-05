import argparse
import os
import pickle
import torch
import numpy as np
import torchvision

import sys
sys.path.append(".")
sys.path.append("..")

from configs.paths_config import edit_paths, model_paths
from editing.styleclip.global_direction import StyleCLIPGlobalDirection
from editing.styleclip.model import Generator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_dir", type=str, default="./experiment",
                        help="Path to inference results with `latents.npy` saved here (obtained with inference.py).")
    parser.add_argument("--weight_deltas_path", type=str, default="./weight_deltas",
                        help="Root path holding all weight deltas (obtained by running inference.py).")
    parser.add_argument('--n_images', type=int, default=None,
                        help="Maximum number of images to edit. If None, edit all images.")
    parser.add_argument("--neutral_text", type=str, default="face with hair")
    parser.add_argument("--target_text", type=str, default="face with long hair")
    parser.add_argument("--stylegan_weights", type=str, default=model_paths["stylegan_ffhq"])
    parser.add_argument("--stylegan_size", type=int, default=1024)
    parser.add_argument("--stylegan_truncation", type=int, default=1.)
    parser.add_argument("--stylegan_truncation_mean", type=int, default=4096)
    parser.add_argument("--beta_min", type=float, default=0.11)
    parser.add_argument("--beta_max", type=float, default=0.16)
    parser.add_argument("--num_betas", type=int, default=5)
    parser.add_argument("--alpha_min", type=float, default=-5)
    parser.add_argument("--alpha_max", type=float, default=5)
    parser.add_argument("--num_alphas", type=int, default=11)
    parser.add_argument("--delta_i_c", type=str, default=edit_paths["styleclip"]["delta_i_c"],
                        help="path to file containing delta_i_c")
    parser.add_argument("--s_statistics", type=str, default=edit_paths["styleclip"]["s_statistics"],
                        help="path to file containing s statistics")
    parser.add_argument("--text_prompt_templates", default=edit_paths["styleclip"]["templates"])
    args = parser.parse_args()
    return args


def load_direction_calculator(args):#构建一个StyleCLIPGlobalDirection对象
    delta_i_c = torch.from_numpy(np.load(args.delta_i_c)).float().cuda()#[6048,512]
    with open(args.s_statistics, "rb") as channels_statistics:
        _, s_std = pickle.load(channels_statistics)#s_std是一个含26个元素的列表，每个元素是一个tensor，shape=[512]
        s_std = [torch.from_numpy(s_i).float().cuda() for s_i in s_std]#放到cuda上
    with open(args.text_prompt_templates, "r") as templates:
        text_prompt_templates = templates.readlines()
    global_direction_calculator = StyleCLIPGlobalDirection(delta_i_c, s_std, text_prompt_templates)
    return global_direction_calculator


def load_stylegan_generator(args): # 载入stylegan2的生成器，带参数的
    stylegan_model = Generator(args.stylegan_size, 512, 8, channel_multiplier=2).cuda()
    checkpoint = torch.load(args.stylegan_weights)
    stylegan_model.load_state_dict(checkpoint['g_ema'])
    return stylegan_model


def run():
    args = parse_args()
    stylegan_model = load_stylegan_generator(args)#载入stylegan2的生成器，带参数的
    global_direction_calculator = load_direction_calculator(args)#构建一个StyleCLIPGlobalDirection对象（用到了editing/styleclip/global_directions里面的3个东西）
    # load latents obtained via inference # 载入inference.py得到的latents.npy
    latents = np.load(os.path.join(args.exp_dir, 'latents.npy'), allow_pickle=True).item()#它是一个dict，key是图片名，value是latent，latent是[18,512]
    # prepare output directory
    args.output_path = os.path.join(args.exp_dir, "styleclip_edits", f"{args.neutral_text}_to_{args.target_text}")#args.neutral_text是'a face'，args.target_text是'a face with a beard'
    os.makedirs(args.output_path, exist_ok=True)
    # edit all images
    for idx, (image_name, latent) in enumerate(latents.items()):#对每一张图片进行编辑
        if args.n_images is not None and idx >= args.n_images:
            break
        edit_image(image_name, latent, stylegan_model, global_direction_calculator, args)#输入图片名称，latent[18,512]（含有图片的内容）,stylegan2的生成器，StyleCLIPGlobalDirection对象，参数args对图片进行编辑


def edit_image(image_name, latent, stylegan_model, global_direction_calculator, args):
    print(f'Editing {image_name}')

    latent_code = torch.from_numpy(latent).cuda()
    truncation = 1
    mean_latent = None
    input_is_latent = True
    latent_code_i = latent_code.unsqueeze(0)#[1,18,512]

    weight_deltas = np.load(os.path.join(args.weight_deltas_path, image_name.split(".")[0] + ".npy"), allow_pickle=True) #载入在infernece.py中得到的weight_deltas，专门对当前图片的权重增量
    weight_deltas = [torch.from_numpy(w).cuda() if w is not None else None for w in weight_deltas]

    with torch.no_grad():

        source_im, _, latent_code_s = stylegan_model([latent_code_i],#[1,18,512]# latent_codes_s是一个含有26个元素的列表，每个元素[1,1,512,1,1]，是latent_code_i经过全连接层的仿射变换得到的（当然18个latent会有重复利用的才构成了26个）
                                                     input_is_latent=input_is_latent,# True
                                                     randomize_noise=False,
                                                     return_latents=True,
                                                     truncation=truncation, #1
                                                     truncation_latent=mean_latent,#None
                                                     weights_deltas=weight_deltas)

    alphas = np.linspace(args.alpha_min, args.alpha_max, args.num_alphas)#-5,5,等间隔的生成11个数[-5. -4. -3. -2. -1.  0.  1.  2.  3.  4.  5.]
    betas = np.linspace(args.beta_min, args.beta_max, args.num_betas)#0.11,0.16,等间隔的生成5个数[0.11  0.1225  0.135   0.1475  0.16  ]
    results = []
    for beta in betas:#s_i可以看做W+空间的在i层的潜在代码，b_i是根据text获得的全局方向，alpha是在全局方向上的步长（我自己想的，还需看论文确认）；beta其实对应着阈值，小于beta的值会被置为0，使用更高的阈值可实现更多的解缠操作，但同时操作的视觉效果会降低
        direction = global_direction_calculator.get_delta_s(args.neutral_text, args.target_text, beta)##是一个含26个元素的list，每个元素是[1,1,512,1,1]的样子，15个512，3个256，3个128，3个64，2个32，只有11个512和1个256里面有数（非to_rgb层），其他都是0
        edited_latent_code_s = [[s_i + alpha * b_i for s_i, b_i in zip(latent_code_s, direction)] for alpha in alphas]#latent_code_s是含26个元素的list，里面是仿射变换之后的style；direction也是含26个元素的list，里面的东西我还不好解释;edited_latent_code_s是含11个元素的list,每个元素是含26个元素的list，每个元素是[1,1,512,1,1]的样子tensor
        edited_latent_code_s = [torch.cat([edited_latent_code_s[i][j] for i in range(args.num_alphas)])
                                for j in range(len(edited_latent_code_s[0]))]#含26个元素的list，每个元素是[11,1,512,1,1]的tensor
        for b in range(0, edited_latent_code_s[0].shape[0]):#range(0,11)
            edited_latent_code_s_batch = [s_i[b:b + 1] for s_i in edited_latent_code_s]#含26个元素，取出[11,1,512,1,1]中的第b个元素，变成[1,1,512,1,1]的tensor
            with torch.no_grad():
                edited_image, _, _ = stylegan_model([edited_latent_code_s_batch],#这是编辑后的潜在代码，输出重建的图片
                                                    input_is_stylespace=True,#这个是True说明输入的style已经经过仿射变换到达W+空间了，无需再进行全连接层的仿射变换
                                                    randomize_noise=False,
                                                    return_latents=True,
                                                    weights_deltas=weight_deltas)
                results.append(edited_image)#edited_image[1,3,1024,1024]根据文本编辑后的图片

    results = torch.cat(results)#阈值beta有5个，扰动幅度alpha有11个，所以一共有55个结果，每个结果是[1,3,1024,1024]的图片
    torchvision.utils.save_image(results, f"{args.output_path}/{image_name.split('.')[0]}.jpg",
                                 normalize=True, range=(-1, 1), padding=0, nrow=args.num_alphas)


if __name__ == "__main__":
    run()
