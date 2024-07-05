import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

import sys
sys.path.append(".")
sys.path.append("..")

from configs import data_configs
from datasets.inference_dataset import InferenceDataset
from editing.face_editor import FaceEditor
from options.test_options import TestOptions
from utils.common import tensor2im
from utils.inference_utils import run_inversion
from utils.model_utils import load_model


def run():
    test_opts = TestOptions().parse()

    out_path_results = os.path.join(test_opts.exp_dir, 'editing_results')
    out_path_coupled = os.path.join(test_opts.exp_dir, 'editing_coupled')

    os.makedirs(out_path_results, exist_ok=True)
    os.makedirs(out_path_coupled, exist_ok=True)

    # update test options with options used during training
    net, opts = load_model(test_opts.checkpoint_path, update_opts=test_opts)#载入hyperStyle的模型和测试的参数

    print('Loading dataset for {}'.format(opts.dataset_type))#载入ffhq_hypernet
    dataset_args = data_configs.DATASETS[opts.dataset_type]
    transforms_dict = dataset_args['transforms'](opts).get_transforms()
    dataset = InferenceDataset(root=opts.data_path,#测试照片的路径
                               transform=transforms_dict['transform_inference'],
                               opts=opts)
    dataloader = DataLoader(dataset,
                            batch_size=opts.test_batch_size,
                            shuffle=False,
                            num_workers=int(opts.test_workers),
                            drop_last=False)

    if opts.n_images is None:
        opts.n_images = len(dataset)

    latent_editor = FaceEditor(net.decoder)# net.decoder是hyperStyle的decoder，实际是stylegan2的生成器，这里面还有interfacegan_directions，包括age,smile,pose等方向pt文件，他们是[1,512]大小的向量

    global_i = 0
    for input_batch in tqdm(dataloader): #[4,3,256,256]

        if global_i >= opts.n_images:
            break

        with torch.no_grad():
            input_cuda = input_batch.cuda().float()#[4,3,256,256]
            result_batch = run_on_batch(input_cuda, net, latent_editor, opts)#输入数据、hyperStyle的模型，潜在代码编辑器和测试参数，返回的是长度为4的list，每个元素又是一个list，里面是字典，key是inversion，age，pose，smile，value是PIL图像

        resize_amount = (256, 256) if opts.resize_outputs else (opts.output_size, opts.output_size)
        for i in range(input_batch.shape[0]):#对每一张输入图像

            im_path = dataset.paths[global_i]
            results = result_batch[i]

            inversion = results.pop('inversion')#这是根据输入图像经过迭代微调stylegan2的生成器得到的重建图像，没有经过编辑
            input_im = tensor2im(input_batch[i])#真实的输入图像

            all_edit_results = []
            for edit_name, edit_res in results.items():#现在results里面只有age,smile,pose三个key
                # set the input image
                res = np.array(input_im.resize(resize_amount))#现在res里面是输入图像
                # set the inversion
                res = np.concatenate([res, np.array(inversion.resize(resize_amount))], axis=1)#再拼接上重建图像
                # add editing results side-by-side
                for result in edit_res:
                    res = np.concatenate([res, np.array(result.resize(resize_amount))], axis=1)#再拼接上10个编辑后的图像
                res_im = Image.fromarray(res) #转为PIL图像，依次是输入图像，重建图像，10个编辑后的图像
                all_edit_results.append(res_im)

                edit_save_dir = os.path.join(out_path_results, edit_name)
                os.makedirs(edit_save_dir, exist_ok=True)
                res_im.save(os.path.join(edit_save_dir, os.path.basename(im_path)))#保存起来

            # save final concatenated result of all edits #保存age，smile，pose三个方向的编辑结果拼接起来的图
            coupled_res = np.concatenate(all_edit_results, axis=0)
            im_save_path = os.path.join(out_path_coupled, os.path.basename(im_path))
            Image.fromarray(coupled_res).save(im_save_path)
            global_i += 1


def run_on_batch(inputs, net, latent_editor, opts):
    y_hat, _, weights_deltas, codes = run_inversion(inputs, net, opts)#输出三次微调stylegan2的生成器得到的重建图像、三次微调的权重增量和潜在代码（应该是w_endoder直接出来的，与微调无关）
    edit_directions = opts.edit_directions.split(',')#['age', 'smile', 'pose']
    # store all results for each sample, split by the edit direction
    results = {idx: {'inversion': tensor2im(y_hat[idx])} for idx in range(len(inputs))}#results是一个字典，key是0，1，2，3，value是一个字典，key是inversion，value是tensor2im(y_hat[idx])是PIL图像
    for edit_direction in edit_directions:#对于每一个编辑方向
        edit_res = latent_editor.apply_interfacegan(latents=codes,#w_encoder的输出[4,18,512]
                                                    weights_deltas=weights_deltas,#26个元素的列表，对应stylegan2的26层的权重
                                                    direction=edit_direction, #string,'age'或 'smile'或 'pose'
                                                    factor_range=(-1 * opts.factor_range, opts.factor_range))#(-5,5)，进行10种程度的编辑
        # store the results for each sample
        for idx, sample_res in edit_res.items():
            results[idx][edit_direction] = sample_res
    return results #长度为4的list,每个元素又是一个list，里面是字典，key是inversion，age，pose，smile，value是PIL图像


if __name__ == '__main__':
    run()
