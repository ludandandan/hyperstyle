import copy
import clip
import torch

from editing.styleclip.stylespace_utils import features_channels_to_s


class StyleCLIPGlobalDirection:

    def __init__(self, delta_i_c, s_std, text_prompts_templates):
        super(StyleCLIPGlobalDirection, self).__init__()
        self.delta_i_c = delta_i_c#[6048,512]，存在fs3.npy里的
        self.s_std = s_std#26个元素的list，存在S_mean_std里面，[512],...[64],[32],[32]
        self.text_prompts_templates = text_prompts_templates#79个文本模板的list，存在templates里面
        self.clip_model, _ = clip.load("ViT-B/32", device="cuda")

    def get_delta_s(self, neutral_text, target_text, beta):#'a face', 'a face with a beard',0.11或0.1225  0.135   0.1475  0.16 
        delta_i = self.get_delta_i([target_text, neutral_text]).float()# target在前，neutral在后，获得target和neutral的text_features的标准化差异向量[512]，长度为1
        r_c = torch.matmul(self.delta_i_c, delta_i)#[6048,512]和[512]相乘，得到[6048]，6048个值
        delta_s = copy.copy(r_c)#[6048]
        channels_to_zero = torch.abs(r_c) < beta#[6048]，小于beta的值为True，大于等于beta的值为False
        delta_s[channels_to_zero] = 0#把小于beta的值置为0
        max_channel_value = torch.abs(delta_s).max()#找到delta_s的绝对值的最大值
        if max_channel_value > 0:
            delta_s /= max_channel_value
        direction = features_channels_to_s(delta_s, self.s_std)#delta_s[6048]，s_std是26个元素的list
        return direction#是一个含26个元素的list，每个元素是[1,1,512,1,1]的样子，15个512，3个256，3个128，3个64，2个32，只有11个512和1个256里面有数（非to_rgb层），其他都是0

    def get_delta_i(self, text_prompts):#['a face with a beard', 'a face']，获得target和neutral的text_features的标准化差异向量
        text_features = self._get_averaged_text_features(text_prompts)#[2,512]
        delta_t = text_features[0] - text_features[1]#目标-中性，[512]，差异向量
        delta_i = delta_t / torch.norm(delta_t)#标准化差异向量，使得向量长度为1
        return delta_i

    def _get_averaged_text_features(self, text_prompts):#['a face with a beard', 'a face']，使用clip对79个含text_prompts进行文本编码并求平均以获得target和neutral的text_features，多个求平均比较稳定一点，每个是[512]
        with torch.no_grad():
            text_features_list = []
            for text_prompt in text_prompts:
                formatted_text_prompts = [template.format(text_prompt) for template in self.text_prompts_templates]  # format with class，把'a face with a beard'填充在模板中的{}里面
                formatted_text_prompts = clip.tokenize(formatted_text_prompts).cuda()  # tokenize，返回给定输入字符串的tokenized表示[79,77]
                text_embeddings = self.clip_model.encode_text(formatted_text_prompts)  # embed with text encoder [79,512]将文本token编码成512维的向量
                text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)#沿着最后一个维度求范数，然后除以这个范数，使得每个向量的范数为1[79,512]
                text_embedding = text_embeddings.mean(dim=0)#[512],就是79个数求平均值，得到一个平均的向量
                text_embedding /= text_embedding.norm()#使得这个向量的范数为1，归一化
                text_features_list.append(text_embedding)
            text_features = torch.stack(text_features_list, dim=1).cuda()#text_features_list是一个有2个元素的list，每个元素[512],text_features[512,2]
        return text_features.t()#[2,512]，分别是target和neutral的text_features
