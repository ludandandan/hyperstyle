import torch


def get_average_image(net, opts):
    avg_image = net(net.latent_avg.unsqueeze(0),#net是e4e，将存储的平均潜在编码，经过stylegan2生成器得到平均图像
                    input_code=True,
                    randomize_noise=False,
                    return_latents=False,
                    average_code=True)[0]
    avg_image = avg_image.to('cuda').float().detach() #[3,256,256]
    if "cars" in opts.dataset_type:
        avg_image = avg_image[:, 32:224, :]
    return avg_image


def run_on_batch(inputs, net, opts): #每个batch经过2次迭代，得到重建图像和潜在编码（都是使用Restyle e4e的编码器和解码器构建潜在编码和生成重建图像）
    avg_image = get_average_image(net, opts) #得到平均图像[3,256,256]
    y_hat, latent = None, None
    for iter in range(opts.n_iters_per_batch):#opts.n_iters_per_batch=2
        if iter == 0: #第一次迭代
            avg_image_for_batch = avg_image.unsqueeze(0).repeat(inputs.shape[0], 1, 1, 1)#[4,3,256,256]
            x_input = torch.cat([inputs, avg_image_for_batch], dim=1) #将原始输入和平均图像拼接在一起
        else:
            x_input = torch.cat([inputs, y_hat], dim=1) #将原始输入和重建图像拼接在一起

        y_hat, latent = net.forward(x_input,#[4,6,256,256]#输入图和平均图或重建图拼接的经过net的编码器生成潜在编码（潜在编码加上平均潜在编码或者前一次的潜在编码作为新的潜在编码用于后续），然后再经过解码器（stylegan2的生成器）得到重建图像，同时返回潜在编码
                                    latent=latent, #一开始是None
                                    randomize_noise=False,
                                    return_latents=True,
                                    resize=opts.resize_outputs)

        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    return y_hat, latent
