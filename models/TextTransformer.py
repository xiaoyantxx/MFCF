import torch
import torch.nn as nn
from timm.models.layers import DropPath
from torch.autograd import Variable


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_size, dropout, sequence_length):
        super(PositionalEncoding, self).__init__()
        # 实例化dropout层
        self.dpot = nn.Dropout(p=dropout)
        # 初始化位置编码矩阵
        pe = torch.zeros(sequence_length, embedding_size)
        # 初始化绝对位置矩阵
        # position矩阵size为(max_len,1)
        position = torch.arange(0, sequence_length).unsqueeze(1)
        # 将绝对位置矩阵和位置编码矩阵特征融合
        # 定义一个变换矩阵 跳跃式初始化
        div_term = torch.exp(torch.arange(0, embedding_size, 2) * -(math.log(10000) / embedding_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 将二维张量扩充成三维张量
        pe = pe.unsqueeze(0)
        # 把pe位置编码矩阵注册成模型的buffer
        # 我们把它认为是对模型效果有帮助的，但是却不是模型结构中超参数或者参数，不需要说着优化步骤进行更新的增益
        # 注册之后我们就可以在模型保存后重新加载时盒模型结构与参数已通被加载
        self.register_buffer('pe', pe)

    def forward(self, x):  # (sequence_length, embedding_size)
        text_input = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dpot(text_input)


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
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv1d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):  # [8,128,100]
        x = self.fc1(x.permute(0, 2, 1)).permute(0, 2, 1)
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

        self.to_q = nn.Linear(dim, nh_kd)
        self.to_k = nn.Linear(dim, nh_kd)
        self.to_v = nn.Linear(dim, self.dh)

        self.proj = torch.nn.Sequential(nn.Linear(self.dh, dim))

    def forward(self, x):  # [8,128,100]
        B, C, N = x.shape
        qq = self.to_q(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, self.num_heads, self.key_dim, N).permute(0, 1, 3, 2)   # [8,16,100,16]
        kk = self.to_k(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, self.num_heads, self.key_dim, N)  # [8,16,16,100]
        vv = self.to_v(x.permute(0, 2, 1)).permute(0, 2, 1).reshape(B, self.num_heads, self.d, N).permute(0, 1, 3, 2)   # [8,16,100,4*16]

        attn = torch.matmul(qq, kk)/self.scale  # [8,16,100,100]
        attn = attn.softmax(dim=-1)  # [8,16,100,100]

        xx = torch.matmul(attn, vv)  # [8,16,100,4*16]

        xx = xx.permute(0, 1, 3, 2).reshape(B, self.dh, N)  # [8,16*4*16,100]
        xx = self.proj(xx.permute(0, 2, 1)).permute(0, 2, 1)  # [8,128,100]
        return xx


# dim:块的输入和输出维度 num_heads：自注意力机制中的头数  mlp_ratio：MLP（多层感知机）层的隐藏层维度相对于输入维度的比例
# drop:丢失概率 act_layer:激活函数的类型 norm_layer：规范化层的类型 window_size：自注意力机制的窗口大小
class TextBlock(nn.Module):
    def __init__(self, dim=256, num_heads=16,  mlp_ratio=4., qkv_bias=False, drop=0.2, attn_drop=0.,
                 drop_path=0., act_layer=nn.ReLU6, norm_layer=nn.BatchNorm1d, window_size=8):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, key_dim=16, num_heads=num_heads)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):  # 8 128 10 10
        B, C, H, W = x.shape
        x = x.view(B, C, -1)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, C, H, W)
        return x


