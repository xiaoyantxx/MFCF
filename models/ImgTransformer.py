import torch
import torch.nn as nn
from timm.models.layers import DropPath


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d, bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# self attention    dim:输入特征维度  key_dim:自注意力的键的维度  num_heads：自注意力中的注意头数  attn_ratio:在计算注意力的过程中，将‘key_dim’扩展到d维度 默认为4
class Attention(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = ConvBN(dim, nh_kd, 1)
        self.to_k = ConvBN(dim, nh_kd, 1)
        self.to_v = ConvBN(dim, self.dh, 1)

        self.proj = torch.nn.Sequential(ConvBN(self.dh, dim, 1))

    def forward(self, x):  # x (B,N,C)
        B, C, H, W = x.shape
        qq = self.to_q(x).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(x).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(x).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)/self.scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


class CrossAttention(nn.Module):
    def __init__(self, dim=128, hid_dim=64):
        super(CrossAttention, self).__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.scale = hid_dim ** -0.5
        self.img_q = ConvBN(dim, hid_dim, 1)
        self.img_k = ConvBN(dim, hid_dim, 1)
        self.text_k = nn.Linear(dim, hid_dim)
        self.text_q = nn.Linear(dim, hid_dim)

    def forward(self, input_i, input_t):
        # torch.Size([8, 128, 14, 14])  torch.Size([8, 128, 10, 10])
        B, C, _, _ = input_i.shape
        # 线性映射
        i_q = self.img_q(input_i).reshape(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 196, 64)
        t_k = self.text_k(input_t.view(B, C, -1).permute(0, 2, 1))  # (8, 100, 64)
        t_v = input_t.view(B, C, -1).permute(0, 2, 1)  # torch.Size([8, 100, 128])
        # 计算注意力权重
        iscores = torch.matmul(i_q, t_k.transpose(1, 2))  # torch.Size([8, 196, 100])
        attentions_i = torch.softmax(iscores/self.scale, dim=-1)  # torch.Size([8, 196, 100])
        # 使用注意力权重来调整输入表示
        output_i = torch.matmul(attentions_i, t_v)   # torch.Size([8, 196, 100]) torch.Size([8, 100, 128])

        # 线性映射
        t_q = self.text_q(input_t.view(B, C, -1).permute(0, 2, 1)).permute(0, 2, 1).reshape(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 100, 64)
        i_k = self.img_k(input_i).view(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 196, 64)
        i_v = input_i.view(B, C, -1).permute(0, 2, 1)  # torch.Size([8, 196, 128])
        # 计算注意力权重
        tscores = torch.matmul(t_q, i_k.transpose(1, 2))  # torch.Size([8, 100,196])
        attentions_t = torch.softmax(tscores / self.scale, dim=-1)  # torch.Size([8, 100,196])
        # 使用注意力权重来调整输入表示
        output_t = torch.matmul(attentions_t, i_v)  # torch.Size([8, 100, 128])
        return output_i, output_t  # torch.Size([8, 196, 128]) torch.Size([8, 100, 128])


# 没有多头
class ImgAttention(nn.Module):
    def __init__(self, dim=128, hid_dim=128):
        super(ImgAttention, self).__init__()
        self.dim = dim
        self.hid_dim = hid_dim
        self.scale = hid_dim ** -0.5
        self.img_q = ConvBN(dim, hid_dim, 1)
        self.img_k = ConvBN(dim, hid_dim, 1)

    def forward(self, input_all, input_reg):
        # torch.Size([8, 128, 14, 14])  torch.Size([8, 128, 14, 14])
        B, C, H, W = input_all.shape
        # qkv
        i_q = self.img_q(input_all).reshape(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 196, 128)
        i_k = self.img_k(input_reg).reshape(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 196, 128)
        i_v = self.img_k(input_reg).reshape(B, self.hid_dim, -1).permute(0, 2, 1)  # (8, 196, 128)
        # 计算注意力权重
        iscores = torch.matmul(i_q, i_k.transpose(1, 2))  # torch.Size([8, 196, 196])
        attentions_i = torch.softmax(iscores/self.scale, dim=-1)  # torch.Size([8, 196, 196])
        # 使用注意力权重来调整输入表示
        output_i = torch.matmul(attentions_i, i_v)   # torch.Size([8, 196, 128])

        output_i = output_i.permute(0, 2, 1).reshape(B, C, H, W)

        return output_i  # torch.Size([8, 128, 14, 14])


class ImgAttention1(torch.nn.Module):
    def __init__(self, dim, key_dim, num_heads,
                 attn_ratio=4,
                 activation=None,):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads  # num_head key_dim
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio

        self.to_q = ConvBN(dim, nh_kd, 1)
        self.to_k = ConvBN(dim, nh_kd, 1)
        self.to_v = ConvBN(dim, self.dh, 1)

        self.proj = torch.nn.Sequential(ConvBN(self.dh, dim, 1))

    def forward(self, all, part):  # x (B,N,C)
        B, C, H, W = all.shape
        qq = self.to_q(all).reshape(B, self.num_heads, self.key_dim, H * W).permute(0, 1, 3, 2)
        kk = self.to_k(part).reshape(B, self.num_heads, self.key_dim, H * W)
        vv = self.to_v(part).reshape(B, self.num_heads, self.d, H * W).permute(0, 1, 3, 2)

        attn = torch.matmul(qq, kk)/self.scale
        attn = attn.softmax(dim=-1)  # dim = k

        xx = torch.matmul(attn, vv)

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, H, W)
        xx = self.proj(xx)
        return xx


# dim:块的输入和输出维度 num_heads：自注意力机制中的头数  mlp_ratio：MLP（多层感知机）层的隐藏层维度相对于输入维度的比例
# drop:丢失概率 act_layer:激活函数的类型 norm_layer：规范化层的类型 window_size：自注意力机制的窗口大小
class ImgBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.2, attn_drop=0.,
                 drop_path=0.5, act_layer=nn.ReLU6, norm_layer=nn.BatchNorm2d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, key_dim=16, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):

        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


img_atten = torch.randn(8, 128, 14, 14)
text_atten = torch.randn(8, 128, 10, 10)
cross_attn = CrossAttention()
img_cross, text_cross = cross_attn(img_atten, text_atten)  # [8, 196, 128]