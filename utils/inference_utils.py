import torch


def run_inversion(inputs, net, opts, return_intermediate_results=False):##返回两次微调stylegan2的生成器得到的重建图像，潜在编码和两次微调累积的权重增量
    y_hat, latent, weights_deltas, codes = None, None, None, None

    if return_intermediate_results:
        results_batch = {idx: [] for idx in range(inputs.shape[0])}
        results_latent = {idx: [] for idx in range(inputs.shape[0])}
        results_deltas = {idx: [] for idx in range(inputs.shape[0])}
    else:
        results_batch, results_latent, results_deltas = None, None, None

    for iter in range(opts.n_iters_per_batch): # 2次迭代
        y_hat, latent, weights_deltas, codes, _ = net.forward(inputs, #这里的net是hyperstyle模型，input[4,3,256,256]，这里是将输入给到hyperstyle模型，然后得到模型预测的对当前输入stylegan2模型各层参数的增量，还有stylgegan2根据增量微调后得到的重建图像，这里的latent和codes是w_encoder得到的潜在编码
                                                              y_hat=y_hat,
                                                              codes=codes,
                                                              weights_deltas=weights_deltas,
                                                              return_latents=True,
                                                              resize=opts.resize_outputs,
                                                              randomize_noise=False,
                                                              return_weight_deltas_and_codes=True)

        if "cars" in opts.dataset_type:
            if opts.resize_outputs:
                y_hat = y_hat[:, :, 32:224, :]
            else:
                y_hat = y_hat[:, :, 64:448, :]

        if return_intermediate_results:
            store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas)

        # resize input to 256 before feeding into next iteration
        if "cars" in opts.dataset_type:
            y_hat = torch.nn.AdaptiveAvgPool2d((192, 256))(y_hat)
        else:
            y_hat = net.face_pool(y_hat)

    if return_intermediate_results:
        return results_batch, results_latent, results_deltas
    return y_hat, latent, weights_deltas, codes#返回两次微调stylegan2的生成器得到的重建图像，潜在编码和两次微调累积的权重增量


def store_intermediate_results(results_batch, results_latent, results_deltas, y_hat, latent, weights_deltas):
    for idx in range(y_hat.shape[0]):
        results_batch[idx].append(y_hat[idx])
        results_latent[idx].append(latent[idx].cpu().numpy())
        results_deltas[idx].append([w[idx].cpu().numpy() if w is not None else None for w in weights_deltas])
