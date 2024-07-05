import os
import matplotlib
import matplotlib.pyplot as plt

from datasets.dataset_fetcher import DatasetFetcher

matplotlib.use('Agg')

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from utils import common, train_utils
from criteria import id_loss, moco_loss
from configs import data_configs
from datasets.images_dataset import ImagesDataset
from datasets.latents_images_dataset import LatentsImagesDataset
from criteria.lpips.lpips import LPIPS
from models.hyperstyle import HyperStyle
from training.ranger import Ranger


class Coach:
    def __init__(self, opts):
        self.opts = opts

        self.global_step = 0

        self.device = 'cuda:0'
        self.opts.device = self.device

        # Initialize network，里面有decoder（stylegan2的生成器）、hypernet（resnet34）、latent_avg（[18,512]）、w_encoder（e4e编码器），opt，face_pool
        self.net = HyperStyle(self.opts).to(self.device)

        # Estimate latent_avg via dense sampling if latent_avg is not available
        if self.net.latent_avg is None: #[18,512]
            self.net.latent_avg = self.net.decoder.mean_latent(int(1e5))[0].detach()

        # Initialize loss 初始化损失，就是定义用哪些损失
        if self.opts.id_lambda > 0 and self.opts.moco_lambda > 0: #如果id损失的乘数因子大于0且moco特征损失的乘数因子大于0同时存在，是不允许的
            raise ValueError('Both ID and MoCo loss have lambdas > 0! Please select only one to have non-zero lambda!')
        self.mse_loss = nn.MSELoss().to(self.device).eval() #像素损失
        if self.opts.lpips_lambda > 0: # 如果LPIPS损失的乘数因子大于0，则定义LPIPS损失
            self.lpips_loss = LPIPS(net_type='alex').to(self.device).eval()
        if self.opts.id_lambda > 0: # 如果ID损失的乘数因子大于0，则定义ID损失
            self.id_loss = id_loss.IDLoss(opts).to(self.device).eval()
        if self.opts.moco_lambda > 0: # 如果moco损失的乘数因子大于0，则定义moco损失
            self.moco_loss = moco_loss.MocoLoss()

        # Initialize optimizer
        self.optimizer = self.configure_optimizers()#配置优化器，这里只优化hypernet的参数，不优化decoder的参数（也就是StyleGAN生成器）

        # Initialize dataset 加载数据集
        self.train_dataset, self.test_dataset = self.configure_datasets()
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=self.opts.batch_size,
                                           shuffle=True,
                                           num_workers=int(self.opts.workers),
                                           drop_last=True)
        self.test_dataloader = DataLoader(self.test_dataset,
                                          batch_size=self.opts.test_batch_size,
                                          shuffle=False,
                                          num_workers=int(self.opts.test_workers),
                                          drop_last=True)

        # Initialize logger
        log_dir = os.path.join(opts.exp_dir, 'logs')
        os.makedirs(log_dir, exist_ok=True)
        self.logger = SummaryWriter(log_dir=log_dir)

        # Initialize checkpoint dir
        self.checkpoint_dir = os.path.join(opts.exp_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.best_val_loss = None
        if self.opts.save_interval is None:
            self.opts.save_interval = self.opts.max_steps

    def parse_batch(self, batch):
        x, y, y_hat, latents = None, None, None, None
        if isinstance(self.train_dataset, ImagesDataset):
            x, y = batch #x是source，y是target
            x, y = x.to(self.device).float(), y.to(self.device).float()
        elif isinstance(self.train_dataset, LatentsImagesDataset):
            y_hat, y, latents = batch  # source (inversion), target, and latent code
            y_hat, y, latents = y_hat.to(self.device).float(), y.to(self.device).float(), latents.to(self.device)
            x = y
        else:
            raise ValueError("Unsupported dataset type")
        return x, y, y_hat, latents #x是source，y是target，y_hat是None，latents是None

    def perform_forward_on_batch(self, batch, train=False):
        latent, weights_deltas, w_inversion, initial_inversion = None, None, None, None #每个batch都会将weights_deltas设置为None
        cur_loss_dict, id_logs = None, None
        x, y, y_hat, codes = self.parse_batch(batch)#解析之后，x是source[8,3,256,256]，y是target[8,3,256,256]，y_hat是None，codes是None
        y_hats = {idx: [] for idx in range(x.shape[0])}#y_hats是一个字典，key是0-7，value是一个列表,现在里面是空列表
        for iter in range(self.opts.n_iters_per_batch):#每批图像迭代n_iters_per_batch次来累积计算weights_deltas
            if iter > 0 and train:#如果不是第一次迭代，且是训练模式
                weights_deltas = [w.clone().detach().requires_grad_(True) if w is not None else w #先创建一个副本，detach是从当前计算图中分离出w的副本，这样不会保留之前计算的梯度信息，requires_grad_(True)是允许对这个新的w计算梯度
                                  for w in weights_deltas] #weights_deltas一开始是None
                y_hat = y_hat.clone().detach().requires_grad_(True)#y_hat一开始是None，clone是创建一个副本，detach是从当前计算图中分离出y_hat的副本，这样不会保留之前计算的梯度信息，requires_grad_(True)是允许对这个新的y_hat计算梯度
            y_hat, latent, weights_deltas, codes, w_inversion = self.net.forward(x, #前向传播，x是source[8,3,256,256]
                                                                                 y_hat=y_hat, #一开始是None
                                                                                 codes=codes, #一开始是None
                                                                                 weights_deltas=weights_deltas, #一开始是None
                                                                                 return_latents=True,
                                                                                 randomize_noise=False,
                                                                                 return_weight_deltas_and_codes=True,
                                                                                 resize=True)
            if iter == 0:
                initial_inversion = w_inversion#初始inversion是stylegan2的原始权重计算得到的重建图像
            if "cars" in self.opts.dataset_type:
                y_hat = y_hat[:, :, 32:224, :]
            loss, cur_loss_dict, id_logs = self.calc_loss(x=x, #源图像[8,3,256,256]，这里原来竟然写的是x=y，我改成x=x
                                                          y=y, #目标图像[8,3,256,256]
                                                          y_hat=y_hat, #stylegan2的权重经过调整后的重建图像[8,3,256,256]
                                                          latent=latent, #潜在变量，还是w_encoder得到的[8,18,512]
                                                          weights_deltas=weights_deltas) #本次SharedWeightsHyperNetResNet的forward函数得到的权重增量,是一个26个元素的list
            if train:
                loss.backward() #反向传播，计算梯度

            # store intermediate outputs
            for idx in range(x.shape[0]): #存储重建图像的中间结果，包括图像和id_logs中的重建和目标之间的点积，越大越好
                y_hats[idx].append([y_hat[idx].detach().cpu(), id_logs[idx]['diff_target']])
        return x, y, y_hats, cur_loss_dict, id_logs, initial_inversion

    def train(self):
        self.net.train()#设置为训练模式
        while self.global_step < self.opts.max_steps:# self.opts.max_steps=500000，self.global_step是一个计数器，每次训练一步，self.global_step+1，看起来是对一个数据集训练500000个batch，而不是500000个epoch
            for batch_idx, batch in enumerate(self.train_dataloader): #batch中含有2组值，一组是source，一组是target（但是我看大家的source和target配置的路径都一样，都是训练照片路径，是不是也可以source是训练照片，target是训练草图）

                self.optimizer.zero_grad() #梯度清零,batch[0]是[8,3,256,256]，batch[1]是[8,3,256,256]

                x, y, y_hat, loss_dict, id_logs, w_inversion = self.perform_forward_on_batch(batch, train=True)
                self.optimizer.step()#之前累积的梯度更新参数，这里的参数是hypernet的参数，也就是SharedWeightsHyperNetResNet类中的参数

                # Logging related
                if self.global_step % self.opts.image_interval == 0 or (self.global_step < 1000 and self.global_step % 25 == 0): #每100个batch或者前1000次每25次记录一次
                    self.parse_and_log_images(id_logs, x, y, y_hat, w_inversion, title='images/train')#x源图像，y目标图像，y_hat是stylegan2微调后的重建图像，w_inversion是stylegan2的原始权重计算得到的重建图像

                if self.global_step % self.opts.board_interval == 0:#每50个epoch打印和记录一次
                    self.print_metrics(loss_dict, prefix='train')
                    self.log_metrics(loss_dict, prefix='train')

                # Validation related
                val_loss_dict = None
                if self.global_step % self.opts.val_interval == 0 or self.global_step == self.opts.max_steps:#每5000个batch或者到最后一个500000个batch的时候验证一次
                    val_loss_dict = self.validate()#执行验证
                    if val_loss_dict and (self.best_val_loss is None or val_loss_dict['loss'] < self.best_val_loss):#保存最好的模型
                        self.best_val_loss = val_loss_dict['loss']
                        self.checkpoint_me(val_loss_dict, is_best=True)

                if self.global_step > 0 and (self.global_step % self.opts.save_interval == 0 or self.global_step == self.opts.max_steps):#每10000次保存一下模型，或者到最后一次的时候保存一次
                    if val_loss_dict is not None:
                        self.checkpoint_me(val_loss_dict, is_best=False)#保存模型
                    else:
                        self.checkpoint_me(loss_dict, is_best=False)

                if self.global_step == self.opts.max_steps: #如果达到最大步数，就结束训练
                    print('OMG, finished training!')
                    break

                self.global_step += 1

    def validate(self):
        self.net.eval()#网络设置为评估模式
        agg_loss_dict = []
        for batch_idx, batch in enumerate(self.test_dataloader):
            if self.opts.max_val_batches is not None and batch_idx > self.opts.max_val_batches:#只做150个batch的验证
                break
            with torch.no_grad():
                x, y, y_hat, cur_loss_dict, id_logs, w_inversion = self.perform_forward_on_batch(batch)#返回源图像、目标图像、微调后的stylegan2得到的重建图像、当前的损失、身份id logs、stylegan2原始权重计算得到的重建图像
            agg_loss_dict.append(cur_loss_dict)

            # Logging related
            self.parse_and_log_images(id_logs, x, y, y_hat, w_inversion,
                                      title='images/test', subscript='{:04d}'.format(batch_idx))

            # For first step just do sanity test on small amount of data 对于第一步，只对少量数据进行测试
            if self.global_step == 0 and batch_idx >= 4:
                self.net.train() #设置为训练模式
                return None  # Do not log, inaccurate in first batch

        loss_dict = train_utils.aggregate_loss_dict(agg_loss_dict)
        self.log_metrics(loss_dict, prefix='test')
        self.print_metrics(loss_dict, prefix='test')

        self.net.train() #设置为训练模式
        return loss_dict

    def checkpoint_me(self, loss_dict, is_best):
        save_name = 'best_model.pt' if is_best else 'iteration_{}.pt'.format(self.global_step)
        save_dict = self.__get_save_dict()
        checkpoint_path = os.path.join(self.checkpoint_dir, save_name)
        torch.save(save_dict, checkpoint_path)
        with open(os.path.join(self.checkpoint_dir, 'timestamp.txt'), 'a') as f:
            if is_best:
                f.write('**Best**: Step - {}, Loss - {:.3f} \n{}\n'.format(self.global_step, self.best_val_loss, loss_dict))
            else:
                f.write('Step - {}, \n{}\n'.format(self.global_step, loss_dict))

    def configure_optimizers(self):
        params = list(self.net.hypernet.parameters())
        if self.opts.train_decoder:#如果训练decoder，就把decoder的参数也加进去【但是默认不训练，也就是不训练StyleGAN2的生成器】
            params += list(self.net.decoder.parameters())
        if self.opts.optim_name == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.opts.learning_rate)
        else: #使用Ranger优化器（Rctified Adam，一种改进版的Adam优化器）
            optimizer = Ranger(params, lr=self.opts.learning_rate)
        return optimizer

    def configure_datasets(self):
        if self.opts.dataset_type not in data_configs.DATASETS.keys():
            raise Exception('{} is not a valid dataset_type'.format(self.opts.dataset_type))
        print('Loading dataset for {}'.format(self.opts.dataset_type))
        dataset_args = data_configs.DATASETS[self.opts.dataset_type]
        transforms_dict = dataset_args['transforms'](self.opts).get_transforms()
        train_dataset, test_dataset = DatasetFetcher().get_dataset(self.opts, dataset_args, transforms_dict)
        print("Number of training samples: {}".format(len(train_dataset)))
        print("Number of test samples: {}".format(len(test_dataset)))
        return train_dataset, test_dataset

    def calc_loss(self, x, y, y_hat, latent, weights_deltas):
        loss_dict = {}
        loss = 0.0
        id_logs = None

        if "cars" in self.opts.dataset_type:
            y_hat_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
            y_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(y)
            x_resized = torch.nn.AdaptiveAvgPool2d((192, 256))(x)
        else:
            y_hat_resized = self.net.face_pool(y_hat) #[8,3,256,256]
            y_resized = self.net.face_pool(y)# [8,3,256,256]
            x_resized = self.net.face_pool(x)

        if self.opts.id_lambda > 0: #如果id损失的乘数因子大于0 ，使用id损失
            loss_id, sim_improvement, id_logs = self.id_loss(y_hat_resized, y_resized, x_resized)
            loss_dict['loss_id'] = float(loss_id)
            loss_dict['id_improve'] = float(sim_improvement)
            loss = loss_id * self.opts.id_lambda
        if self.opts.l2_lambda > 0:
            loss_l2 = F.mse_loss(y_hat, y) #计算重建与目标之间的像素损失
            loss_dict['loss_l2'] = float(loss_l2)
            loss += loss_l2 * self.opts.l2_lambda
        if self.opts.lpips_lambda > 0:
            loss_lpips = self.lpips_loss(y_hat_resized, y_resized)
            loss_dict['loss_lpips'] = float(loss_lpips)
            loss += loss_lpips * self.opts.lpips_lambda
        if self.opts.moco_lambda > 0:
            loss_moco, sim_improvement, id_logs = self.moco_loss(y_hat_resized, y_resized, x_resized)
            loss_dict['loss_moco'] = float(loss_moco)
            loss_dict['id_improve'] = float(sim_improvement)
            loss += loss_moco * self.opts.moco_lambda

        loss_dict['loss'] = float(loss)#总损失
        return loss, loss_dict, id_logs

    def log_metrics(self, metrics_dict, prefix):
        for key, value in metrics_dict.items():
            self.logger.add_scalar('{}/{}'.format(prefix, key), value, self.global_step)

    def print_metrics(self, metrics_dict, prefix):
        print('Metrics for {}, step {}'.format(prefix, self.global_step))
        for key, value in metrics_dict.items():
            print('\t{} = '.format(key), value)

    def parse_and_log_images(self, id_logs, x, y, y_hat, w_inversion, title, subscript=None, display_count=2):
        im_data = []
        display_count = min(x.shape[0], display_count)# 展示2张图
        for i in range(display_count):
            if type(y_hat) == dict:
                output_face = [#里面是图片+diff_target（重建图像与目标图像特征之间的点积，越大越好）
                    [common.tensor2im(y_hat[i][iter_idx][0]), y_hat[i][iter_idx][1]]
                    for iter_idx in range(len(y_hat[i]))
                ]
            else:
                output_face = [common.tensor2im(y_hat[i])]
            cur_im_data = {
                'input_face': common.tensor2im(x[i]),
                'target_face': common.tensor2im(y[i]),
                'output_face': output_face,#长度是5，因为要迭代5次，每次都有一个重建图像和diff_target
                'w_inversion': common.tensor2im(w_inversion[i])#stylegan2的原始权重计算得到的重建图像（这里其实和源相似）
            }
            if id_logs is not None:
                for key in id_logs[i]:#id_logs是一个字典，key是0-7，value是一个字典，里面有diff_target（重建图像与目标图像），diff_input（重建图像与输入图像）和diff_views（目标图像与输入图像，这个应该是不变的）
                    cur_im_data[key] = id_logs[i][key]
            im_data.append(cur_im_data)
        self.log_images(title, im_data=im_data, subscript=subscript)

    def log_images(self, name, im_data, subscript=None, log_latest=False):
        fig = common.vis_faces(im_data)
        step = self.global_step
        if log_latest:
            step = 0
        if subscript:
            path = os.path.join(self.logger.log_dir, name, '{}_{:04d}.jpg'.format(subscript, step))
        else:
            path = os.path.join(self.logger.log_dir, name, '{:04d}.jpg'.format(step))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fig.savefig(path)
        plt.close(fig)

    def __get_save_dict(self):
        save_dict = {
            'state_dict': {k: v for k, v in self.net.state_dict().items() if 'w_encoder' not in k},#保存模型参数，但是不保存w_encoder的参数，那就是只保存hypernet和decoder的参数
            'opts': vars(self.opts),
            'latent_avg': self.net.latent_avg,
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss
        }
        return save_dict

    def __load_train_checkpoint(self, checkpoint):
        print('Loading previous training data...')
        self.global_step = checkpoint['global_step'] + 1
        self.best_val_loss = checkpoint['best_val_loss']
        # self.net.load_state_dict(checkpoint['state_dict'])
        print(f'Resuming training from step: {self.global_step}')
        print(f'Current best validation loss: {self.best_val_loss}')
