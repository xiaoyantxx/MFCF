import numpy as np
from torch import nn
import torch.nn.functional as F
from models import resnet, Arpn
from models.ImgTransformer import ImgBlock, CrossAttention
from models.TextTransformer import TextBlock
from models.Arpn import ProposalBLock
from models.anchors import generate_default_anchor_maps


class mfcf(nn.Module):
    def __init__(self, num_classes, vocab_size, embedding_size, topN, device):
        super(mfcf, self).__init__()
        self.dim = embedding_size
        self.topN = topN
        self.device = device
        # resnet
        self.pretrained_model = resnet.resnet50(pretrained=True)
        self.pretrained_model.avgpool = nn.AdaptiveAvgPool2d(1)
        self.pretrained_model.fc = nn.Linear(512 * 4, num_classes)
        # part feature
        self.proposal_block = ProposalBLock()
        self.rpn_avg = nn.AvgPool2d(16)
        self.rpn_conv = nn.Conv2d(6 * 3, self.dim, kernel_size=1)
        self.concat_net = nn.Linear(2048 * (4 + 1), num_classes)  # nn.Linear(2048 * (CAT_NUM + 1), num_classes)
        self.partcls_net = nn.Linear(512 * 4, num_classes)
        _, edge_anchors, _ = generate_default_anchor_maps()
        self.pad_side = 224
        self.edge_anchors = (edge_anchors + 224).astype(np.int)
        # img layer
        self.img_conv = nn.Conv2d(2048, self.dim, kernel_size=1)
        # text Embedding layer
        self.embedding_layer = nn.Embedding(vocab_size, self.dim)
        # attn layer
        self.imgtransformer_encoder = ImgBlock(dim=self.dim)
        self.texttransformer_encoder = TextBlock(dim=self.dim)
        self.att_conv = nn.Conv2d(256, self.dim, kernel_size=1)
        self.cross_attn = CrossAttention()
        self.avgpool2 = nn.AdaptiveAvgPool2d(1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, 45)

    def forward(self, input_i, input_t):
        batch = input_t.size(0)
        resnet_out, res_feature, feature = self.pretrained_model(input_i)
        res_logits = resnet_out

        # part_feature
        x_pad = F.pad(input_i, (self.pad_side, self.pad_side, self.pad_side, self.pad_side), mode='constant', value=0)
        # we will reshape rpn to shape: batch * nb_anchor
        rpn_score = self.proposal_block(res_feature.detach())
        top_n_coordinates, part_imgs, top_n_index, top_n_prob = \
            Arpn.RpnBlock(rpn_score, self.edge_anchors, x_pad, self.device, self.topN)
        # part_imgs: torch.Size([48(8*6), 3, 224, 224])
        _, rpn_part, part_features = self.pretrained_model(part_imgs.detach())  # [48, 2048, 7,7]
        part_logits = self.partcls_net(part_features).view(batch, self.topN, -1)
        rpn_part = rpn_part.view(batch, self.topN, 2048, 7, 7).permute(0, 2, 1, 3, 4)  # [8, 2048, 6, 7, 7]
        rpn_part = rpn_part[:, :, :4, ...].contiguous()  # [8, 2048, 4, 7, 7]
        rpn_part = rpn_part.reshape(batch, 2048, 14, 14)
        # combine_feature = torch.cat([res_feature, rpn_part], dim=2)
        combine_feature = res_feature + rpn_part

        #  img layer   56 57 58 59   3000词库   62 （5000）
        img_feature = self.img_conv(combine_feature)  # img_feature:[8,128,14,14]
        img_atten = self.imgtransformer_encoder(img_feature)  # img_atten:[8,128,14,14]
        img_avg = self.avgpool2(img_atten).view(batch, -1)  # [8,128]
        img_logits = self.fc(img_avg)

        # text Embedding layer
        text_embedding = self.embedding_layer(input_t).permute(0, 2, 1)  # [8,128,100]
        text_feature = text_embedding.reshape(batch, self.dim, 10, 10)
        text_atten = self.texttransformer_encoder(text_feature)  # text_atten:[8,128,10,10]
        text_avg = self.avgpool2(text_atten).view(batch, -1)  # [8,128]
        text_logits = self.fc(text_avg)

        # cross attention
        img_cross, text_cross = self.cross_attn(img_atten, text_atten)  # [8, 196, 128]  392
        mul_feature = img_cross.permute((0, 2, 1))  # [8, 128, 196] 392
        mul_feature = self.avgpool1(mul_feature).reshape(batch, -1)  # [8, 128]
        mul_logits = self.fc(mul_feature)

        return part_logits, top_n_prob, img_logits, res_logits, text_logits, mul_logits
