# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------

'''
Simple Baselines for Image Restoration

@article{chen2022simple,
  title={Simple Baselines for Image Restoration},
  author={Chen, Liangyu and Chu, Xiaojie and Zhang, Xiangyu and Sun, Jian},
  journal={arXiv preprint arXiv:2204.04676},
  year={2022}
}
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_util import LayerNorm2d
from basicsr.models.archs.local_arch import Local_Base

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

import pywt
from torch.autograd import Function
from torch.autograd import Variable, gradcheck


class GlobalAttention(nn.Module):
    def __init__(self, dim, mode = "Cswin",  heads = 8, dim_head = 64, dropout = 0., patch_size = 4):
        super().__init__()
        if mode == "Cswin":
            self.to_out = Cswin(dim, heads, dim_head, dropout, patch_size)
        elif mode == "CswinPool":
            self.to_out = CswinPool(dim, heads, dim_head, dropout, patch_size)
        elif mode == "CswinPool_fix":
            self.to_out = CswinPool(dim, heads, dim_head, dropout, patch_size, size=8)
        elif mode == "VanillaAttention":
            self.to_out = VanillaAttention(dim, heads=heads)
        elif mode == "LocalAttention":
            self.to_out = LocalAttention(dim, patch_size=8, heads=heads)
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        return self.to_out(x)

class Local(nn.Module):
    def __init__(self, dim, mode = "LocalAttention", split = True, heads = 1):
        super().__init__()
        self.split = split
        if split:
            if mode == "LocalAttention":
                self.to_out_1 = LocalAttention(dim//3,patch_size=8, heads= heads)
                self.to_out_2 = LocalAttention(dim//3,patch_size=8, heads= heads)
                self.to_out_3 = LocalAttention(dim//3,patch_size=8, heads= heads)
            elif mode == "LocalCNN":
                self.to_out_1 = LocalCNN(dim//3)
                self.to_out_2 = LocalCNN(dim//3)
                self.to_out_3 = LocalCNN(dim//3)
            elif mode == "LocalConvNeXt":
                self.to_out_1 = LocalConvNeXt(dim//3)
                self.to_out_2 = LocalConvNeXt(dim//3)
                self.to_out_3 = LocalConvNeXt(dim//3)
            else:
                self.to_out_1 = nn.Identity()
                self.to_out_2 = nn.Identity()
                self.to_out_3 = nn.Identity()
        else:
            if mode == "LocalAttention":
                self.to_out = LocalAttention(dim, patch_size=8, heads=heads)
            elif mode == "LocalCNN":
                self.to_out = LocalCNN(dim)
            elif mode == "LocalConvNeXt":
                self.to_out = LocalConvNeXt(dim)
            else:
                self.to_out = nn.Identity()

    def forward(self, x):
        if self.split:
            x = x.chunk(3, dim=1)
            x_out_1 = self.to_out_1(x[0])
            x_out_2 = self.to_out_2(x[1])
            x_out_3 = self.to_out_3(x[2])
            x_out = torch.cat([x_out_1, x_out_2, x_out_3], dim=1)
        else:
            x_out = self.to_out(x)
        return x_out

class VanillaAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_qkvL = nn.Conv2d(dim, inner_dim * 4, 1, bias = False)

        self.to_out = nn.Sequential(
                nn.Conv2d(inner_dim, dim, 1),
                nn.Dropout(dropout)
                )

    def forward(self, fmap):
        shape = fmap.shape
        b, n, x, y, h = *shape, self.heads
        
        #fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)
        qkvL = self.to_qkvL(fmap).chunk(4, dim = 1)
        q, k, v, L = map(lambda t: rearrange(t, 'b (h d) x y -> (b h) (x y) d', h = h), qkvL)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v) + L
        out = rearrange (out, '(b h) (x y) d -> b (h d) x y', h=h, x=x, y=y)
        out = self.to_out(out)
        return out

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkvL = self.to_qkvL(fmap).chunk(4, dim = 1)
        # self.to_qkvL = nn.Conv2d(self.dim, self.inner_dim * 4, 1, bias = False)
        flops += N * (2 * self.dim * 1 * 1 - 1) * self.inner_dim * 4 
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        flops += 2 * self.heads * N * (self.inner_dim // self.heads) * N
        # out = self.to_out(out)
        flops += 2 * N * self.inner_dim * self.dim * 1
        return flops/2


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

    def flops(self, N):
        flops = 0
        if self.data_format == "channels_last":
            return flops
        elif self.data_format == "channels_first":
            # not counted
            return flops

class CswinPool(nn.Module):
    def __init__(self, dim,  heads = 8, dim_head = 32, attn_drop=0., patch_size = 4, size = 0):
        super().__init__()
        self.dim = dim
        self.split_size = patch_size
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.size = 0

        if size is not 0:
            self.to_qv_vertical = nn.Conv2d(dim//2, dim, 1, bias = False)
            self.to_qv_horizontal = nn.Conv2d(dim//2, dim,1, bias = False)
            self.AdaptiveAvgPool1d = nn.AdaptiveAvgPool1d(size)
            self.size = size
        else:
            self.to_qv_vertical = nn.Conv2d(dim//2, dim, (1,2),(1,2),padding=(0, 0), bias = False)
            self.to_qv_horizontal = nn.Conv2d(dim//2, dim, (2,1),(2,1),padding=(0,0), bias = False)
        self.to_k = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_lepe = nn.Conv2d(dim, dim, 1, bias = False)
        self.proj = nn.Conv2d(dim, dim , 1, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, fmap):
        shape, p = fmap.shape, self.split_size
        b, n, x, y, h = *shape, self.num_heads


        qv_horizontal = self.to_qv_horizontal(fmap.chunk(2, dim =1)[0])
        qv_vertical = self.to_qv_vertical(fmap.chunk(2, dim =1)[1])
        if self.size is not 0:
            qv_vertical = rearrange(self.AdaptiveAvgPool1d(rearrange(qv_vertical, 'B C H W -> (B C) H W')), '(B C) H W -> B C H W', B=b)
            qv_horizontal = rearrange(self.AdaptiveAvgPool1d(rearrange(qv_horizontal, 'B C H W -> (B C) W H')), '(B C) W H -> B C H W', B=b)
            horizontal = (self.size, self.split_size)
            vertical = (self.split_size, self.size)
        else:
            horizontal = (x//2, self.split_size)
            vertical = (self.split_size, y//2)
        k = self.to_k(fmap)
        k_horizontal, k_vertical = k.chunk(2, dim=1)
        #(b, c, x, p1, y, p2)->(b x, y, p1, p2, c) -> (b x y)(p1 p2) c
        qv_horizontal = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = horizontal[0], p2 = horizontal[1], h = self.num_heads), qv_horizontal.chunk(2, dim = 1))
        qv_vertical = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = vertical[0], p2 = vertical[1], h = self.num_heads), qv_vertical.chunk(2, dim=1))
        q_vertical, v_vertical = qv_vertical
        q_horizontal, v_horizontal = qv_horizontal

        v_horizontal_lepe, v_vertical_lepe = self.to_lepe(fmap).chunk(2, dim=1)

        v_vertical_lepe, k_vertical = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = vertical[0], p2 = y, h = self.num_heads), (v_vertical_lepe, k_vertical))
        v_horizontal_lepe, k_horizontal = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = x, p2 = horizontal[1], h = self.num_heads), (v_horizontal_lepe, k_horizontal))

        horizontal = (x, self.split_size)
        vertical = (self.split_size, y)
        qkv_horizontal = (q_horizontal, k_horizontal, v_horizontal, v_horizontal_lepe, horizontal)
        qkv_vertical = (q_vertical, k_vertical,  v_vertical, v_vertical_lepe, vertical)
        
        qkv = (qkv_horizontal, qkv_vertical)
        v_out = []
        for q, k, v, lepe, (H, W) in qkv:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)
            v = (attn.transpose(-2,-1) @ v)+ lepe

            v = rearrange(v, '(b x y) h (p1 p2) d -> b (h d) (x p1) (y p2)',x =x//H, y = y//W, h = self.num_heads, p1 = H, p2 = W, d = self.dim // self.num_heads //2)

            v_out.append(v)
        fmap = fmap + self.proj(torch.cat([v_out[0], v_out[1]], dim = 1))
        return fmap

class Cswin(nn.Module):
    def __init__(self, dim,  heads = 8, dim_head = 32, attn_drop=0., patch_size = 4):
        super().__init__()
        self.dim = dim
        self.split_size = patch_size
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5

        self.to_qk = nn.Conv2d(dim, dim * 2, 1, bias = False)
        self.to_v = nn.Conv2d(dim, dim, 1, bias = False)
        self.to_v_vertical = nn.Conv2d(dim//2, dim//2, 1, bias = False)
        self.to_v_horizontal = nn.Conv2d(dim//2, dim//2, 1, bias = False)
        self.proj = nn.Conv2d(dim, dim , 1, 1, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, fmap):
        shape, p = fmap.shape, self.split_size
        b, n, x, y, h = *shape, self.num_heads

        horizontal = (x, self.split_size)
        vertical = (self.split_size, y)

        qk = self.to_qk(fmap).chunk(2, dim =1)
        v = self.to_v(fmap)
        #(b, c, x, p1, y, p2)->(b x, y, p1, p2, c) -> (b x y)(p1 p2) c
        qk_horizontal = map(lambda t: rearrange(t.chunk(2, dim=1)[0], 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = horizontal[0], p2 = horizontal[1], h = self.num_heads), qk)
        qk_vertical = map(lambda t: rearrange(t.chunk(2, dim=1)[1], 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = vertical[0], p2 = vertical[1], h = self.num_heads), qk)

        v_vertical_lepe = self.to_v_vertical(v.chunk(2, dim=1)[1])
        v_horizontal_lepe = self.to_v_horizontal(v.chunk(2, dim=1)[0])
        v_vertical, v_horizontal = v.chunk(2, dim =1)[1], v.chunk(2, dim=1)[0]

        v_vertical_lepe, v_vertical = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = vertical[0], p2 = vertical[1], h = self.num_heads), (v_vertical_lepe, v_vertical))
        v_horizontal_lepe, v_horizontal = map(lambda t: rearrange(t, 'b (h d) (x p1) (y p2) -> (b x y) h (p1 p2) d', p1 = horizontal[0], p2 = horizontal[1], h = self.num_heads), (v_horizontal, v_horizontal))

        qkv_horizontal = (qk_horizontal, v_horizontal, v_horizontal_lepe, horizontal)
        qkv_vertical = (qk_vertical, v_vertical, v_vertical_lepe, vertical)
        
        qkv = (qkv_horizontal, qkv_vertical)
        v_out = []
        for (q, k), v, lepe, (H, W) in qkv:
            q = q * self.scale
            attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
            attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
            attn = self.attn_drop(attn)

            v = (attn @ v) + lepe

            v = rearrange(v, '(b x y) h (p1 p2) d -> b (h d) (x p1) (y p2)',x =x//H, y = y//W, h = self.num_heads, p1 = H, p2 = W, d = self.dim // self.num_heads //2)

            v_out.append(v)
        fmap = fmap + self.proj(torch.cat([v_out[0], v_out[1]], dim = 1))
        return fmap

    def flops(self, H, W):
        flops = 0
        N = H * W
        # qkv
        flops += N * self.dim * self.dim * 3
        # q = q * self.scale
        # self.scale = head_dim ** -0.5
        flops += N * self.num_heads * N
        # attn = (q @ k.transpose(-2, -1)) 
        flops += self.num_heads * N * (self.dim // self.num_heads)
        # v = (attn @ v) + lepe
        flops += self. num_heads * N * N * (self.dim // self.num_heads) + N
        # fmap = fmap + self.proj(torch.cat([v_out[0], v_out[1]], dim = 1))
        flops += N * self.dim * self.dim
        return flops

class LocalConvNeXt(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dim = dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = rearrange(x, 'N C H W -> N H W C')
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = rearrange(x, 'N H W C -> N C H W')

        x = input + self.drop_path(x)
        return x
    
    def flops(self, H, W):
        N = H * W
        # Hout = (H - 7 + 2 * 3) + 1
        # Wout = (W - 7 + 2 * 3) + 1
        flops = 0
        # x = self.dwconv(x)
        # self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        flops += 2 * N * self.dim * self.dim * 7 * 7 / self.dim
        # x = self.pwconv1(x)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)
        flops += 2 * self.dim * 4 * self.dim
        # x = self.pwconv2(x)
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        flops += 2 * 4 * self.dim * self.dim
        return flops/2

class LocalCNN(nn.Module):
    def __init__(self, dim, DW_Expand=2):
        super().__init__()
        self.dim = dim
        dw_channel = DW_Expand * dim
        self.dw_channel = dw_channel
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.gelu = nn.GELU()
    def forward(self, fmap):
        fmap = self.conv1(fmap)
        fmap = self.conv2(fmap)
        fmap = self.gelu(fmap)
        fmap = self.conv3(fmap)
        return fmap

    def flops(self, N):
        flops = 0
        # fmap = self.conv1(fmap)
        # self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.dim * self.dw_channel * 1 * 1
        # fmap = self.conv2(fmap)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        flops += 2 * N * self.dw_channel * self.dw_channel * 3 * 3 / self.dw_channel
        # fmap = self.conv3(fmap)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.dw_channel * self.dim * 1 * 1
        return flops/2

class LocalAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., patch_size = 4):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)
        self.dim = dim
        self.inner_dim = inner_dim
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.patch_size = patch_size

        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
                nn.Conv2d(inner_dim, dim, 1),
                nn.Dropout(dropout)
                )

    def forward(self, fmap):
        shape, p = fmap.shape, self.patch_size
        b, n, x, y, h = *shape, self.heads
        if x < p or y < p:
            p = x
        x, y = map(lambda t: t // p, (x, y))
        
        fmap = rearrange(fmap, 'b c (x p1) (y p2) -> (b x y) c p1 p2', p1 = p, p2 = p)
        qkv = self.to_qkv(fmap).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h d) p1 p2 -> (b h) (p1 p2) d', h = h), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange (out, '(b x y h) (p1 p2) d -> b (h d) (x p1) (y p2)', h=h, x=x, y=y, p1=p, p2=p)
        out = self.to_out(out)
        return out
   
    def flops(self, N):
        flops = 0
        # qkv = self.to_qkv(fmap).chunk(3, dim = 1)
        # self.to_qkv = nn.Conv2d(dim, inner_dim * 3, 1, bias = False)
        flops += N * self.dim * self.inner_dim * 3
        # dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        flops += self.heads * N * (self.dim // self.heads) * N * self.scale
        # out = self.to_out(out)
        # self.to_out = nn.Sequential( nn.Conv2d(inner_dim, dim, 1),nn.Dropout(dropout))
        flops += N * self.inner_dim * self.dim
        return flops

class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dim_head = 32, bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.inner_dim = num_heads * dim_head
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=bias)
        self.proj_drop = nn.Dropout(proj_drop)


    def forward(self, x):
        b,c,h,w = x.shape

        qkv = self.qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (head c) h w -> b head c (h w)', head = self.num_heads), qkv)
        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

    def flops(self, N):
        flops = 0
        # qkv = self.to_qkv(fmap).chunk(3, dim = 1)
        # self.qkv = nn.Conv2d(dim, self.inner_dim * 3, 1, bias=False)
        flops += N * (2 * self.dim - 1) * self.inner_dim * 3   
        # attn = (q @ k.transpose(-2, -1)) * self.temperature
        flops +=  2 * self.num_heads * N * (self.dim // self.num_heads) * N * self.temperature  
        # out = (attn @ v)
        flops += 2 * self. num_heads * N * N * (self.dim // self.num_heads)
        # out = self.project_out(out)
        # self.project_out = nn.Conv2d(self.inner_dim, dim, kernel_size=1, bias=False)
        flops += N * (2 * self.inner_dim -1) * self.dim
        return flops/2   

class Mix(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
    
    def forward(self, x1, x2):
        return x1 + x2 * self.beta

class RecursiveBlock(nn.Module):
    def __init__(self, c, G, L, split=True, Max_Level = 4):
        super().__init__()
        self.level = Max_Level
        self.DWT = nn.ModuleList()
        self.Attention = nn.ModuleList()
        self.IDWT = nn.ModuleList()
        self.Local = nn.ModuleList()
        self.Mix = nn.ModuleList()
        for i in range(Max_Level):
            self.Attention.append(ChannelAttention(dim = c*4, num_heads = int(4 * 2**(3 - Max_Level))))
            self.Local.append(Local(dim = c*3, split=split,mode=L, heads= int(3 * 2**(3 - Max_Level))))
            self.Mix.append(Mix(dim = c))
            self.DWT.append(DWT_2D(wave='haar'))
            self.IDWT.append(IDWT_2D(wave='haar'))
        self.Mix.append(Mix(dim=c))
        self.Global = GlobalAttention(c, G)

    def forward(self, inp):
        xll_input = inp
        xll_list, xlh_list, xhl_list, xhh_list = [], [], [], []

        for i in range(self.level):
            x_ll, x_lh, x_hl, x_hh = self.DWT[i](xll_input).chunk(4, dim=1)
            xll_list.append(x_ll)
            xlh_list.append(x_lh)
            xhl_list.append(x_hl)
            xhh_list.append(x_hh)
            xll_input = x_ll
        
        if self.level != 0:
            x_up = xll_list[-1]
        else:
            x_up = inp
        x_up = self.Global(x_up)
        for i in range(self.level - 1, -1, -1):
            x_ll, x_lh, x_hl, x_hh = xll_list[i], xlh_list[i], xhl_list[i], xhh_list[i]
            x_ll = self.Mix[i](x_ll, x_up)
            # local first
            H = self.Local[i](torch.cat([x_lh, x_hl, x_hh], dim=1))
            out = self.Attention[i](torch.cat([x_ll, H], dim=1))

            x_up = self.IDWT[i](out)
        
        out = self.Mix[0](inp, x_up)
        return out

    def flops(self, N):
        flops = 0
        return flops

class NAFBlock(nn.Module):
    def __init__(self, i, c, G, L, S , DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.dim = c
        self.dw_channel = dw_channel
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
#        self.sca = nn.Sequential(
#            nn.AdaptiveAvgPool2d(1),
#            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
#                      groups=1, bias=True),
#        )

        # SimpleGate
        self.sg1 = RecursiveBlock(dw_channel, G, L,split = S, Max_Level=i)

        ffn_channel = FFN_Expand * c
        self.ffn_channel = ffn_channel
        self.sg2 = nn.GELU()
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg1(x)
#        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

    def flops(self, H, W):
        flops = 0
        N = H * W
        # self.conv1(x)
        # self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.dim * self.dw_channel
        # x = self.conv2(x)
        # self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel, bias=True)
        flops += N * 2 * self.dw_channel * 3 * 3  * self.dw_channel / self.dw_channel
        # x = self.conv3(x)
        # self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.dw_channel * self.dim
        # y = inp + x * self.beta
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        flops += N + N 
        # x = self.conv4(self.norm2(y))
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.dim * self.ffn_channel
        # x = self.conv5(x)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        flops += 2 * N * self.ffn_channel * self.dim
        # y + x * self.gamma
        flops += 2 * (N + N)
        return flops/2    

class WaveletNet(nn.Module):

    def __init__(self, G = "VanillaAttention", L = "LocalAttention", S= False, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(len(enc_blk_nums) - i, chan, G =G, L = L, S=S) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(0, chan, G= G, L =L, S=S) for _ in range(middle_blk_num)]
            )

        for i, num in enumerate(dec_blk_nums):
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(i+1, chan, G= G, L= L, S=S) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def flops(self, N):
        flops = 0 
        # x = self.intro(inp)
        # self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1, bias=True)
        flops += N * 2 * 3 * 16 * 3 * 3
        # x = self.ending(x)
        # self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,bias=True)
        flops += N * 2 * 16 * 3 * 3 * 3
        # x = x + inp
        flops += 2 * N
        return flops/2

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class WaveletNetLocal(Local_Base, WaveletNet):
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        WaveletNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)

class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        #return x_ll, x_lh, x_hl, x_hh
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors
            B, C, H, W = ctx.shape
            dx = dx.view(B, 4, -1, H//2, W//2)

            dx = dx.transpose(1,2).reshape(B, -1, H//2, W//2)
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0).repeat(C, 1, 1, 1)
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C)

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            B, C, H, W = ctx.shape
            C = C // 4
            dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters

    def forward(self, x):
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_lh = dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1)
        w_hl = dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1)
        w_hh = dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll
        self.w_lh = self.w_lh
        self.w_hl = self.w_hl
        self.w_hh = self.w_hh
        # self.w_ll = self.w_ll
        # self.w_lh = self.w_lh
        # self.w_hl = self.w_hl
        # self.w_hh = self.w_hh

    def forward(self, x):
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)

if __name__ == '__main__':
    from util_modelsummary import measure_throughput_gpu, get_model_profile

    IMG_LIST = []
    DIM = 48
    IMG_DIM = 3
    PROFILE_IMG_SIZE = (DIM, 256, 256)
    # image test case
    IMG_LIST.append(torch.ones(size = (1, DIM, 256 , 256)))
    IMG_LIST.append(torch.ones(size = (1, DIM, 144 , 144)))

    # net test case
    #GLOBAL = ["Cswin", "CswinPool", "CswinPool_fix", "VanillaAttention", "Identity"]
    GLOBAL = ["Cswin", "CswinPool", "CswinPool_fix"]
    LOCAL = ["LocalAttention", "LocalCNN", "LocalConvNeXt"]
    #GLOBAL = ["Cswin", "CswinPool", "CswinPool_fix", "Identity"]
    #LOCAL = ["LocalAttention", "LocalCNN", "LocalConvNeXt", "Identity"]
    
    OUT = []
    for G in GLOBAL:
        net = GlobalAttention(dim = DIM, mode=G)
        try:
            for Img in IMG_LIST:
                net(Img)
            OUT.append(get_model_profile(net, PROFILE_IMG_SIZE))
            print("PASS", "Global: ", G)
        except Exception as e:
            print("Global: ", G)
            raise(e)

    for L in LOCAL:
        net = Local(dim = DIM, mode = L, split=True)
        try:
            for Img in IMG_LIST:
                net(Img)
            OUT.append(get_model_profile(net, PROFILE_IMG_SIZE))
            print("PASS", "Local: ", L, "split = True")
        except Exception as e:
            print("Local: ", L, "split = True")
            raise(e)

    for L in LOCAL:
        net = Local(dim = DIM, mode = L, split=False)
        try:
            for Img in IMG_LIST:
                net(Img)
            OUT.append(get_model_profile(net, PROFILE_IMG_SIZE))
            print("PASS", "Local: ", L, "split = False")
        except Exception as e:
            print("Local: ", L, "split = False")
            raise(e)

    OUT.reverse()
    print("Test Shape:", PROFILE_IMG_SIZE)
    print(">"*30, "Global NetWork", "<"*30)
    for G in GLOBAL:
        f, m, p = OUT.pop()
        print(">>>", G, "flops: {:<.4f} [G]".format(f), "params: {:<.4f}[M]".format(p), "activations: {:<.4f} [M]".format(m))
    print(">"*30, "Local  NetWork", "<"*30)
    print(">"*10, "Split: True")
    for L in LOCAL:
        f, m, p = OUT.pop()
        print(">>>", L, "flops: {:<.4f} [G]".format(f), "params: {:<.4f}[M]".format(p), "activations: {:<.4f} [M]".format(m))
    print(">"*10, "Split: False")
    for L in LOCAL:
        f, m, p = OUT.pop()
        print(">>>", L, "flops: {:<.4f} [G]".format(f), "params: {:<.4f}[M]".format(p), "activations: {:<.4f} [M]".format(m))

    print(">"*30, "WaveletNet NetWork", "<"*30)

    print("="*80)
    img_channel = 3
    width = 32 

    #enc_blks = [2, 2, 4, 8]
    #middle_blk_num = 12
    #dec_blks = [2, 2, 2, 2]

    enc_blks = [4, 6, 6]
    middle_blk_num = 8
    dec_blks = [4, 6, 6]

    S = (True, False)
    for s in S:
        for L in LOCAL:
            for G in GLOBAL:
                try:
                    net = WaveletNet(G = G, L =L, S= s, img_channel=img_channel, width=width, middle_blk_num=middle_blk_num,
                                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
                    flops, activations, params = get_model_profile(net, (3, 256, 256))
                    lantency = measure_throughput_gpu(net, (1, 3, 256, 256))
                    #print("Global:",G,"Local:",L,"Split:",s,"flops:",flops, "macs:",macs,"params:", params)
                    print("Global:",G,"Local:",L,"Split:",s,"flops: {:<.4f} [G]".format(flops), "params: {:<.4f}[M]".format(params), "activations: {:<.4f} [M]".format(activations), "lantency: {:<.4f} [S]".format(lantency))
                except Exception as e:
                    print("Global:",G,"Local:",L,"Split:",s)
