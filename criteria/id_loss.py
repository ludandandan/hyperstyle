import torch
from torch import nn
from configs.paths_config import model_paths
from models.encoders.model_irse import Backbone


class IDLoss(nn.Module):
    def __init__(self, opts):
        super(IDLoss, self).__init__() #加载预训练的resnet50的arcface模型，并将其设置为eval模式
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        self.opts = opts

    def extract_feats(self, x): 
        x = x[:, :, 35:223, 32:220]  # Crop interesting region#裁剪输入图像以保留感兴趣的区域（从 (35, 32) 到 (223, 220)）
        x = self.face_pool(x)#然后使用自适应平均池化调整图像大小，
        x_feats = self.facenet(x)#最后通过预训练的 facenet 模型提取特征。
        return x_feats

    def forward(self, y_hat, y, x):
        n_samples = x.shape[0]
        x_feats = self.extract_feats(x) # 源图像提取特征 [8,512]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there 目标图像提取特征
        y_hat_feats = self.extract_feats(y_hat) #重建图像提取特征
        y_feats = y_feats.detach() #不需要梯度
        loss = 0
        sim_improvement = 0
        id_logs = []
        count = 0
        for i in range(n_samples): #计算每个样本的损失
            diff_target = y_hat_feats[i].dot(y_feats[i]) #计算重建图像和目标图像的特征之间的点积，点积是各元素相乘再相加，与余弦cos有关，两个向量越相似，点积越大
            diff_input = y_hat_feats[i].dot(x_feats[i]) #计算重建图像和源图像的特征之间的点积
            diff_views = y_feats[i].dot(x_feats[i])#计算目标图像和源图像的特征之间的点积
            id_logs.append({'diff_target': float(diff_target),
                            'diff_input': float(diff_input),
                            'diff_views': float(diff_views)})
            loss += 1 - diff_target#计算损失，1减去重建图像和目标图像的特征之间的点积，越小越好
            id_diff = float(diff_target) - float(diff_views)# 重建与目标的相似度 - 源与目标的相似度，得到相似度改进，衡量在身份特征上的改进，应该是越大越好吧，id_diff 为正值，说明生成图像在身份特征上比输入图像更接近目标图像；如果为负值，说明生成图像在身份特征上比输入图像更差。
            sim_improvement += id_diff
            count += 1

        return loss / count, sim_improvement / count, id_logs #返回平均损失，平均相似度改进，id_logs是一个list，每个元素是一个字典，记录了diff_target、diff_input、diff_views
