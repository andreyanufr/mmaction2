#!/usr/bin/env python3

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import math

import torch
import torch.nn as nn
#from timm.models.layers import trunc_normal_
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from collections import OrderedDict

try:
    from .gcvivit import WindowAttentionGlobal, WindowAttention, Mlp, DropPath
except:
    from gcvivit import WindowAttentionGlobal, WindowAttention, Mlp, DropPath

try:
    from ..builder import BACKBONES
except:
    from mmaction.models.builder import BACKBONES

from ptflops import get_model_complexity_info


class SE3D(nn.Module):
    def __init__(self, inp, oup, expansion=0.25):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(oup, int(inp * expansion), bias=False),
            nn.GELU(),
            nn.Linear(int(inp * expansion), oup, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y


class ReduceSize3D(nn.Module):
    def __init__(self, dim,
                 norm_layer=nn.LayerNorm,
                 keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE3D(dim, dim),
            nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
        )
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Conv3d(dim, dim_out, 3, (1, 2, 2), 1, bias=False)
        self.norm2 = norm_layer(dim_out)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        x = x.contiguous()
        x = self.norm1(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = x + self.conv(x)
        x = self.reduction(x).permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm2(x)
        return x


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans=3, dim=96):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, dim, 3, 2, 1)
        self.conv_down = ReduceSize3D(dim=dim, keep_dim=True)

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 4, 1).contiguous()
        x = self.conv_down(x)
        return x


def window_partition3D(x, window_size):
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse3D(windows, window_size, T, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size * T))
    x = windows.view(B, T, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 2, 4, 3, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


def window_partition_temporal(x, window_size):
    B, T, H, W, C = x.shape
    x = x.view(B, T, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 2, 4, 1, 3, 5, 6).contiguous().view(-1, T, window_size * window_size * C)
    return windows


def window_reverse_temporal(windows, window_size, T, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, T, window_size, window_size, -1)
    x = x.permute(0, 3, 1, 4, 2, 5, 6).contiguous().view(B, T, H, W, -1)
    return x


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, d_in: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int = 3, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.l1 = nn.Linear(d_in, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.d_model = d_model
        self.l2 = nn.Linear(d_model, d_in)

    def forward(self, src: Tensor, src_mask: Tensor = None) -> Tensor:
        x = self.l1(src)
        x = self.pos_encoder(x)
        output = self.transformer_encoder(x, src_mask)
        x = self.l2(output)
        return src + x


class GCViTBlock3D(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttentionGlobal,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):
        super().__init__()
        self.window_size = window_size
        self.norm1 = norm_layer(dim)

        self.attn = attention(dim,
                              num_heads=num_heads,
                              window_size=window_size,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)
        self.layer_scale = False
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.layer_scale = True
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        else:
            self.gamma1 = 1.0
            self.gamma2 = 1.0

        self.num_windows = int((input_resolution // window_size) * (input_resolution // window_size))

    def forward(self, x, q_global):
        B, T, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x_windows = window_partition3D(x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        attn_windows = self.attn(x_windows, q_global)
        x = window_reverse3D(attn_windows, self.window_size, T, H, W)
        x = shortcut + self.drop_path(self.gamma1 * x)
        x = x + self.drop_path(self.gamma2 * self.mlp(self.norm2(x)))
        return x


class GCViTBlockTemporal(nn.Module):
    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 attention=WindowAttention,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 ):
        super().__init__()
        self.window_size = window_size
        in_dim = dim * window_size ** 2
        hidden_dim = 128
        self.attn = TransformerModel(in_dim, hidden_dim, num_heads, mlp_ratio * hidden_dim)

    def forward(self, x):
        B, T, H, W, C = x.shape

        x_windows = window_partition_temporal(x, self.window_size)
        attn_windows = self.attn(x_windows)
        x = window_reverse_temporal(attn_windows, self.window_size, T, H, W)

        return x


# class QToGlobal3D(nn.Module):
#     def __init__(self, dim, keep_dim=False):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Conv3d(dim, dim, 3, 1, 1,
#                       groups=dim, bias=False),
#             nn.GELU(),
#             SE3D(dim, dim),
#             nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
#         )

#     def forward(self, x):
#         x = x.contiguous()
#         x = x + self.conv(x)
#         x = x.mean(dim=(-3, -2, -1), keepdim=True)
#         return x


class FeatExtract3D(nn.Module):
    def __init__(self, dim, keep_dim=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(dim, dim, 3, 1, 1,
                      groups=dim, bias=False),
            nn.GELU(),
            SE3D(dim, dim),
            nn.Conv3d(dim, dim, 1, 1, 0, bias=False),
        )
        if not keep_dim:
            self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.keep_dim = keep_dim

    def forward(self, x):
        x = x.contiguous()
        x = x + self.conv(x)
        if not self.keep_dim:
            #print("SHAPE before pool: ", x.shape)
            x = self.pool(x)
            #print("SHAPE after pool: ", x.shape)
        return x

class GlobalQueryGen3D(nn.Module):
    """
    Global query generator based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 num_heads,
                 norm_layer=nn.LayerNorm):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            num_heads: number of heads.
        For instance, repeating log(56/7) = 3 blocks, with input window dimension 56 and output window dimension 7 at
        down-sampling ratio 2. Please check Fig.5 of GC ViT paper for details.
        """

        super().__init__()
        if input_resolution == 56:
            self.to_q_global = nn.Sequential(
                FeatExtract3D(dim, keep_dim=False),
                FeatExtract3D(dim, keep_dim=False),
                FeatExtract3D(dim, keep_dim=False),
            )

        elif input_resolution == 28:
            self.to_q_global = nn.Sequential(
                FeatExtract3D(dim, keep_dim=False),
                FeatExtract3D(dim, keep_dim=False),
            )

        elif input_resolution == 14:

            if window_size == 14:
                self.to_q_global = nn.Sequential(
                    FeatExtract3D(dim, keep_dim=True)
                )

            elif window_size == 7:
                self.to_q_global = nn.Sequential(
                    FeatExtract3D(dim, keep_dim=False)
                )

        elif input_resolution == 7:
            self.to_q_global = nn.Sequential(
                FeatExtract3D(dim, keep_dim=True)
            )
        
        q_global_dim = 7
        if input_resolution == 56:
            q_global_dim = int(input_resolution / 2 ** len(self.to_q_global))
        elif input_resolution == 28:
            q_global_dim = int(input_resolution / 2 ** len(self.to_q_global))
        elif input_resolution == 14:
            if window_size == 14:
                q_global_dim = input_resolution
            elif window_size == 7:
                q_global_dim = int(input_resolution / 2 ** len(self.to_q_global))
        elif input_resolution == 7:
            q_global_dim = input_resolution

        self.norm = norm_layer(dim)
        self.resolution = input_resolution
        self.num_heads = num_heads
        self.N = window_size * window_size
        self.dim_head = torch.div(dim, self.num_heads, rounding_mode='floor')
        self.q_global_dim = q_global_dim

    def forward(self, x):
        # x = _to_channel_last(self.to_q_global(x))
        # B = x.shape[0]
        # x = x.reshape(B, 1, self.N, self.num_heads, self.dim_head).permute(0, 1, 3, 2, 4)
        # return x
        return self.to_q_global(x)


class GCViTLayer3D(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_global=False):
        super().__init__()
        self.blocks = nn.ModuleList([
            GCViTBlock3D(dim=dim,
                         num_heads=num_heads,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attention=WindowAttention if (i % 2 == 0) else WindowAttentionGlobal,
                         drop=drop,
                         attn_drop=attn_drop,
                         drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                         norm_layer=norm_layer,
                         layer_scale=layer_scale,
                         input_resolution=input_resolution)
            for i in range(depth)])
        self.temporal_layer = GCViTBlockTemporal(dim=dim,
                         num_heads=num_heads,
                         window_size=window_size,
                         mlp_ratio=mlp_ratio,
                         qkv_bias=qkv_bias,
                         qk_scale=qk_scale,
                         attention=None,
                         drop=drop,
                         attn_drop=attn_drop,
                         drop_path=None,
                         norm_layer=norm_layer,
                         layer_scale=layer_scale,
                         input_resolution=input_resolution)

        self.downsample = None if not downsample else ReduceSize3D(dim=dim, norm_layer=norm_layer)

        self.q_global_gen = GlobalQueryGen3D(dim, input_resolution, window_size, num_heads)

        self.dim = dim
        self.resolution = input_resolution

    def forward(self, x):
        # q_global = self.to_q_global(x.view(-1,
        #                                    self.dim,
        #                                    -1,
        #                                    self.resolution,
        #                                    self.resolution))
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        #print('SHAPE before :', x.shape)
        q_global = self.q_global_gen(x)
        #print('SHAPE after :', x.shape)
        q_global = torch.mean(q_global, dim=2).squeeze(2)

        q_global = self.q_global_gen.norm(q_global.permute(0, 2, 3, 1))
        q_global = q_global.permute(0, 3, 1, 2)
        
        # q_global = q_global.view(-1,
        #                 self.dim,
        #                 self.q_global_gen_.q_global_dim,
        #                 self.q_global_gen_.q_global_dim)

        x = x.permute(0, 2, 3, 4, 1).contiguous()
        for blk in self.blocks:
            x = blk(x, q_global)
        x = self.temporal_layer(x)
        if self.downsample is None:
            return x
        return self.downsample(x)



@BACKBONES.register_module()
class GCViViT3D(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_global=False,
                 pretrained=None,
                 **kwargs):
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer3D(dim=int(dim * 2 ** i),
                                 depth=depths[i],
                                 num_heads=num_heads[i],
                                 window_size=window_size[i],
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 norm_layer=norm_layer,
                                 downsample=(i < len(depths) - 1),
                                 layer_scale=layer_scale,
                                 input_resolution=int(2 ** (-2 - i) * resolution),
                                 use_global=use_global)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.apply(self._init_weights)

        if pretrained is not None:
            self.load_from2d(pretrained)

        # self.time_attention = WindowAttention(num_features, num_heads=num_heads[0],
        #                                       window_size=window_size[i])

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def load_from2d(self, name):
        state_dict = torch.load(name, map_location='cpu')
        model_state_dict = self.state_dict()
        dict_3d = OrderedDict()
        print("ST: ",  state_dict.keys())
        print("MST: ", model_state_dict.keys())
        for key, val in model_state_dict.items():
            s_key = 'backbone.' + key
            if not s_key in state_dict:
                print('Missed key: ', key)
                continue
            else:
                print('Good key: ', key)
            val_new = state_dict[s_key]
            if len(val.shape) > len(val_new.shape):
                val_new = val_new.unsqueeze(2)
                val_new = val_new.repeat(1, 1, val.shape[2], 1, 1)
                state_dict[key] = val_new
            state_dict[key] = val_new
            dict_3d[s_key] = val_new
        torch.save(dict_3d, name + "_3d")
        self.load_state_dict(state_dict, strict=False)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for level in self.levels:
            x = level(x)
        #
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        #print(x.shape)
        x = self.avgpool(x)
        #print(x.shape)
        x = torch.flatten(x, 1)
        #print(x.shape)
        return x

    def forward(self, x):
        # # shape of the video is [B, C, T, H, W]
        # x = video.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W] -> [B, T, C, H, W]
        # B, T, C, H, W = x.shape
        # assert (T % 2 == 0)
        #
        # x = x.reshape(B * T // 2, 2 * C, H, W)  # tubelet embedding like in https://arxiv.org/pdf/2103.15691.pdf

        x = self.forward_features(x)

        return x


#@BACKBONES.register_module()
class GCViViT3DReccurent(nn.Module):
    def __init__(self,
                 dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm,
                 layer_scale=None,
                 use_global=False,
                 **kwargs):
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed3D(in_chans=in_chans, dim=dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            level = GCViTLayer3D(dim=int(dim * 2 ** i),
                                 depth=depths[i],
                                 num_heads=num_heads[i],
                                 window_size=window_size[i],
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 drop=drop_rate, attn_drop=attn_drop_rate,
                                 drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                 norm_layer=norm_layer,
                                 downsample=(i < len(depths) - 1),
                                 layer_scale=layer_scale,
                                 input_resolution=int(2 ** (-2 - i) * resolution),
                                 use_global=use_global)
            self.levels.append(level)
        self.norm = norm_layer(num_features)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.apply(self._init_weights)

        # self.time_attention = WindowAttention(num_features, num_heads=num_heads[0],
        #                                       window_size=window_size[i])

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x, hiddens):
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i in range(len(self.levels)):
            level, h = self.levels[i], hiddens[i]
            x, h = level(x, h)
            hiddens[i] = h

        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x, hiddens

    def forward(self, video):
        # # shape of the video is [B, C, T, H, W]
        hiddens = [None] * len(self.levels)
        T = video.shape[2]
        predictons = []
        for t in range(T):
            x = video[:, :, t, :, :]
            x, hiddens = self.forward_features(x, hiddens)
            predictons.append(x)
        return torch.stack(predictons, dim=1)


def gc_vit_xxtiny3D(pretrained=False, **kwargs):
    model = GCViViT3D(depths=[3, 3, 3, 3],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 14, 7],
                      dim=32,
                      mlp_ratio=3,
                      drop_path_rate=0.2,
                      **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model

def gc_vit_xxtiny3DRecurrent(pretrained=False, **kwargs):
    model = GCViViT3DReccurent(depths=[3, 3, 3, 3],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 14, 7],
                      dim=32,
                      mlp_ratio=3,
                      drop_path_rate=0.2,
                      **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


def gc_vit_xxtiny3D_112(pretrained=False, **kwargs):
    model = GCViViT3D(depths=[3, 3, 3, 3],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 14, 7],
                      dim=64,
                      mlp_ratio=3,
                      drop_path_rate=0.2,
                      resolution=112,
                      use_global=True,
                      **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


if __name__ == '__main__':
    # model = gc_vit_xxtiny3D_112()
    # x = torch.randn((2, 3, 32, 112, 112))
    model = gc_vit_xxtiny3D()
    x = torch.randn((1, 3, 8, 224, 224))
    with torch.no_grad():
        y = model(x)
    print("Completed. {}".format(y.shape))
    # B, T, C = y.shape
    # y = y.view(B * T, -1)
    # print("Completed. {}".format(y.shape))
    # y = y.view(B, T, -1)
    # print("Completed. {}".format(y.shape))
    #
    macs, params = get_model_complexity_info(model, (3, 16, 224, 224), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    y = model(x)
    loss = torch.mean((y - 1.0)**2)
    loss.backward()

    for name, param in model.named_parameters():
        if param.grad is None:
            print(name)
