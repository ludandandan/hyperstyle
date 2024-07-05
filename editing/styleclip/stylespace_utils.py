import torch

STYLESPACE_DIMENSIONS = [512 for _ in range(15)] + [256, 256, 256] + [128, 128, 128] + [64, 64, 64] + [32, 32]

TORGB_INDICES = list(range(1, len(STYLESPACE_DIMENSIONS), 3))
STYLESPACE_INDICES_WITHOUT_TORGB = [i for i in range(len(STYLESPACE_DIMENSIONS)) if i not in TORGB_INDICES][:11]

def features_channels_to_s(s_without_torgb, s_std):#s_without_torgb[6048]，s_std是26个元素的list
    s = []
    start_index_features = 0
    for c in range(len(STYLESPACE_DIMENSIONS)):#STYLESPACE_DIMENSIONS是stylegan2的26个层的channel数15个512，3个256，3个128，3个64，2个32
        if c in STYLESPACE_INDICES_WITHOUT_TORGB:#不是torgb层[0,2,3,5,6,8,9,11,12,14,15] 前11个非torgb层，10个512，1个256
            end_index_features = start_index_features + STYLESPACE_DIMENSIONS[c]
            s_i = s_without_torgb[start_index_features:end_index_features] * s_std[c]#截取0-512，512-1024，1024-1536，1536-2048，2048-2560，2560-3072，3072-3584，3584-4096，4096-4608，4608-5120，5120-5376
            start_index_features = end_index_features
        else:
            s_i = torch.zeros(STYLESPACE_DIMENSIONS[c]).cuda()#[512]全是0
        s_i = s_i.view(1, 1, -1, 1, 1)#[512]->[1,1,512,1,1]
        s.append(s_i)
    return s#是一个有26个元素的list，除了[0,2,3,5,6,8,9,11,12,14,15]不是0之外，其他都是[1,1,512,1,1]的tensor或者[1,1,256,1,1]的tensor