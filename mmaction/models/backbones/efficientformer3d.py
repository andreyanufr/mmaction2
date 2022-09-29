"""
EfficientFormer
"""
import os
import copy
import torch
import torch.nn as nn
from collections import OrderedDict

from typing import Dict
import itertools
#from ptflops import get_model_complexity_info
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath, trunc_normal_
from timm.models.layers.helpers import to_2tuple, to_3tuple

try:
    from ..builder import BACKBONES
except:
    from mmaction.models.builder import BACKBONES

import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

EfficientFormer_width = {
    'l0': [8, 80, 184, 448],
    'l1': [48, 96, 224, 448],
    'l3': [64, 128, 320, 512],
    'l7': [96, 192, 384, 768],
}

EfficientFormer_depth = {
    'l0': [3, 2, 6, 4],
    'l1': [3, 2, 6, 4],
    'l3': [4, 4, 12, 6],
    'l7': [6, 6, 18, 8],
}


class Attention(torch.nn.Module):
    def __init__(self, dim=384, key_dim=32, num_heads=8,
                 attn_ratio=4,
                 resolution=7):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd * 2
        self.qkv = nn.Linear(dim, h)
        self.proj = nn.Linear(self.dh, dim)

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,N,C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.reshape(B, N, self.num_heads, -1).split([self.key_dim, self.key_dim, self.d], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (
                (q @ k.transpose(-2, -1)) * self.scale
                +
                (self.attention_biases[:, self.attention_bias_idxs]
                 if self.training else self.ab)
        )
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.dh)
        x = self.proj(x)
        return x


def stem(in_chs, out_chs):
    return nn.Sequential(
        nn.Conv3d(in_chs, out_chs // 2, kernel_size=3, stride=2, padding=1),
        nn.BatchNorm3d(out_chs // 2),
        nn.ReLU(),
        nn.Conv3d(out_chs // 2, out_chs, kernel_size=3, stride=(1, 2, 2), padding=1),
        nn.BatchNorm3d(out_chs),
        nn.ReLU(), )


class Embedding(nn.Module):
    """
    Patch Embedding that is implemented by a layer of conv.
    Input: tensor in shape [B, C, H, W]
    Output: tensor in shape [B, C, H/stride, W/stride]
    """

    def __init__(self, patch_size=16, stride=16, padding=0,
                 in_chans=3, embed_dim=768, norm_layer=nn.BatchNorm3d,
                 first=False):
        super().__init__()
        patch_size = to_3tuple(patch_size)
        if first:
            stride = (stride, stride, stride)
        else:
            stride = (max(stride//2, 1), stride, stride)
        padding = to_3tuple(padding)
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return x


class Flat(nn.Module):

    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        return x

class Flat3D(nn.Module):

    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv3d(dim, dim, kernel_size=(5, 1, 1), padding=(2, 0, 0))

    def forward(self, x):
        x = self.conv(x)
        x = x.mean(2)
        x = x.flatten(2).transpose(1, 2)
        return x


class Pooling(nn.Module):
    """
    Implementation of pooling for PoolFormer
    --pool_size: pooling size
    """

    def __init__(self, pool_size=3):
        super().__init__()
        self.pool = nn.AvgPool3d(
            pool_size, stride=1, padding=pool_size // 2, count_include_pad=False)

        self.pool_spatial = nn.AvgPool3d(
            (1, pool_size, pool_size), stride=1, padding=(0, pool_size // 2, pool_size // 2), count_include_pad=False)

    def forward(self, x):
        if x.shape[2] < 3:
            return self.pool_spatial(x) - x
        return self.pool(x) - x


class LinearMlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class Mlp(nn.Module):
    """
    Implementation of MLP with 1*1 convolutions.
    Input: tensor with shape [B, C, H, W]
    """

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv3d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

        self.norm1 = nn.BatchNorm3d(hidden_features)
        self.norm2 = nn.BatchNorm3d(out_features)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)

        x = self.norm1(x)

        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)

        x = self.norm2(x)

        x = self.drop(x)
        return x


class Meta3D(nn.Module):

    def __init__(self, dim, mlp_ratio=4.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):

        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = Attention(dim)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = LinearMlp(in_features=dim, hidden_features=mlp_hidden_dim,
                             act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(0).unsqueeze(0)
                * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(0).unsqueeze(0)
                * self.mlp(self.norm2(x)))

        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Meta4D(nn.Module):

    def __init__(self, dim, pool_size=3, mlp_ratio=4.,
                 act_layer=nn.GELU,
                 drop=0., drop_path=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()

        self.token_mixer = Pooling(pool_size=pool_size)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. \
            else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.layer_scale_2 = nn.Parameter(
                layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        if self.use_layer_scale:

            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.token_mixer(x))
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
                * self.mlp(x))
        else:
            x = x + self.drop_path(self.token_mixer(x))
            x = x + self.drop_path(self.mlp(x))
        return x


def meta_blocks(dim, index, layers,
                pool_size=3, mlp_ratio=4.,
                act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                drop_rate=.0, drop_path_rate=0.,
                use_layer_scale=True, layer_scale_init_value=1e-5, vit_num=1):
    blocks = []
    if index == 3 and vit_num == layers[index]:
        blocks.append(Flat3D(dim))
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (
                block_idx + sum(layers[:index])) / (sum(layers) - 1)
        if index == 3 and layers[index] - block_idx <= vit_num:
            blocks.append(Meta3D(
                dim, mlp_ratio=mlp_ratio,
                act_layer=act_layer, norm_layer=norm_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            add_rnn = True
        else:
            blocks.append(Meta4D(
                dim, pool_size=pool_size, mlp_ratio=mlp_ratio,
                act_layer=act_layer,
                drop=drop_rate, drop_path=block_dpr,
                use_layer_scale=use_layer_scale,
                layer_scale_init_value=layer_scale_init_value,
            ))
            if index == 3 and layers[index] - block_idx - 1 == vit_num:
                blocks.append(Flat3D(dim))

    blocks = nn.Sequential(*blocks)
    return blocks


class EfficientFormer(nn.Module):

    def __init__(self, layers, embed_dims=None,
                 mlp_ratios=4, downsamples=None,
                 pool_size=3,
                 norm_layer=nn.LayerNorm, act_layer=nn.GELU,
                 num_classes=1000,
                 down_patch_size=3, down_stride=2, down_pad=1,
                 drop_rate=0., drop_path_rate=0.,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 fork_feat=False,
                 init_cfg=None,
                 pretrained=None,
                 pretrained_2d=None,
                 vit_num=0,
                 distillation=True,
                 **kwargs):
        super().__init__()

        if not fork_feat:
            self.num_classes = num_classes
        self.fork_feat = fork_feat

        self.patch_embed = stem(3, embed_dims[0])

        network = []
        first_embed = True
        for i in range(len(layers)):
            stage = meta_blocks(embed_dims[i], i, layers,
                                pool_size=pool_size, mlp_ratio=mlp_ratios,
                                act_layer=act_layer, norm_layer=norm_layer,
                                drop_rate=drop_rate,
                                drop_path_rate=drop_path_rate,
                                use_layer_scale=use_layer_scale,
                                layer_scale_init_value=layer_scale_init_value,
                                vit_num=vit_num)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if downsamples[i] or embed_dims[i] != embed_dims[i + 1]:
                # downsampling between two stages
                network.append(
                    Embedding(
                        patch_size=down_patch_size, stride=down_stride,
                        padding=down_pad,
                        in_chans=embed_dims[i], embed_dim=embed_dims[i + 1],
                        first=first_embed
                    )
                )
                first_embed = not first_embed

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        self.apply(self.cls_init_weights)

        self.init_cfg = copy.deepcopy(init_cfg)
        # load pre-trained model
        if self.fork_feat and (
                self.init_cfg is not None or pretrained is not None):
            self.init_weights()

        if pretrained_2d is not None:
            self.load_from2d(pretrained_2d)

    # init for classification
    def cls_init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def load_from2d(self, name):
        state_dict = torch.load(name, map_location='cpu')['model']
        model_state_dict = self.state_dict()
        dict_3d = OrderedDict()
        print("ST: ", state_dict.keys())
        print("MST: ", model_state_dict.keys())
        for key, val in model_state_dict.items():
            s_key = key
            if not s_key in state_dict:
                print('Missed key: ', key)
                continue
            else:
                print('Good key: ', key)
            val_new = state_dict[s_key]
            if len(val.shape) > len(val_new.shape):
                val_new = val_new.unsqueeze(2)
                val_new = val_new.repeat(1, 1, val.shape[2], 1, 1)
            if len(val.shape) > 0 and val.shape[0] < val_new.shape[0]:
                val_new = val_new[:val.shape[0], ...]
            if len(val.shape) > 1 and val.shape[1] < val_new.shape[1]:
                val_new = val_new[:, :val.shape[1], ...]
                state_dict[key] = val_new
            if val_new.shape != val.shape:
                print('Mismatch shape for key: ', key, val.shape, val_new.shape)
            state_dict[key] = val_new
            dict_3d['backbone.' + s_key] = val_new
        torch.save(dict_3d, name + "_3d")
        self.load_state_dict(state_dict, strict=False)

    # # init for mmdetection or mmsegmentation by loading
    # # imagenet pre-trained weights
    # def init_weights(self, pretrained=None):
    #     logger = get_root_logger()
    #     if self.init_cfg is None and pretrained is None:
    #         logger.warn(f'No pre-trained weights for '
    #                     f'{self.__class__.__name__}, '
    #                     f'training start from scratch')
    #         pass
    #     else:
    #         assert 'checkpoint' in self.init_cfg, f'Only support ' \
    #                                               f'specify `Pretrained` in ' \
    #                                               f'`init_cfg` in ' \
    #                                               f'{self.__class__.__name__} '
    #         if self.init_cfg is not None:
    #             ckpt_path = self.init_cfg['checkpoint']
    #         elif pretrained is not None:
    #             ckpt_path = pretrained
    #
    #         ckpt = _load_checkpoint(
    #             ckpt_path, logger=logger, map_location='cpu')
    #         if 'state_dict' in ckpt:
    #             _state_dict = ckpt['state_dict']
    #         elif 'model' in ckpt:
    #             _state_dict = ckpt['model']
    #         else:
    #             _state_dict = ckpt
    #
    #         state_dict = _state_dict
    #         missing_keys, unexpected_keys = \
    #             self.load_state_dict(state_dict, False)

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

    def forward_tokens(self, x):
        outs = []
        for idx, block in enumerate(self.network):
            x = block(x)
            #print(x.shape)
        #     if self.fork_feat and idx in self.out_indices:
        #         norm_layer = getattr(self, f'norm{idx}')
        #         x_out = norm_layer(x)
        #         outs.append(x_out)
        # if self.fork_feat:
        #     return outs
        return x

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.forward_tokens(x)
        if self.fork_feat:
            # otuput features of four stages for dense prediction
            return x
        x = self.norm(x).mean(-2)
        return x
        # if self.dist:
        #     cls_out = self.head(x.mean(-2)), self.dist_head(x.mean(-2))
        #     if not self.training:
        #         cls_out = (cls_out[0] + cls_out[1]) / 2
        # else:
        #     cls_out = self.head(x.mean(-2))
        # for image classification
        #return cls_out

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .95, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'classifier': 'head',
        **kwargs
    }


# @register_model
def efficientformer_l1(pretrained_2d=None, **kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,
        pretrained_2d=pretrained_2d,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

def efficientformer_l0(pretrained_2d=None, **kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l0'],
        embed_dims=EfficientFormer_width['l0'],
        downsamples=[True, True, True, True],
        vit_num=1,
        pretrained_2d=pretrained_2d,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

@BACKBONES.register_module()
def EfficientFormer_l1_3D(**kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

@BACKBONES.register_module()
def EfficientFormer_l0_3D(**kwargs):
    model = EfficientFormer(
        layers=EfficientFormer_depth['l0'],
        embed_dims=EfficientFormer_width['l0'],
        downsamples=[True, True, True, True],
        vit_num=1,
        **kwargs)
    model.default_cfg = _cfg(crop_pct=0.9)
    return model

#
#
# @register_model
# def efficientformer_l3(pretrained=False, **kwargs):
#     model = EfficientFormer(
#         layers=EfficientFormer_depth['l3'],
#         embed_dims=EfficientFormer_width['l3'],
#         downsamples=[True, True, True, True],
#         vit_num=4,
#         **kwargs)
#     model.default_cfg = _cfg(crop_pct=0.9)
#     return model
#
#
# @register_model
# def efficientformer_l7(pretrained=False, **kwargs):
#     model = EfficientFormer(
#         layers=EfficientFormer_depth['l7'],
#         embed_dims=EfficientFormer_width['l7'],
#         downsamples=[True, True, True, True],
#         vit_num=8,
#         **kwargs)
#     model.default_cfg = _cfg(crop_pct=0.9)
#     return model


if __name__ == '__main__':
    # for name in specification:
    #     net = globals()[name](fuse=True, pretrained=True)
    #     net.eval()
    #     net(torch.randn(4, 3, 224, 224))
    #     print(name,
    #           net.FLOPS, 'FLOPs',
    #           sum(p.numel() for p in net.parameters() if p.requires_grad), 'parameters')

    # x = torch.randn(5, 14 * 14, 10)
    # model = AttentionRecurrent(10, 20, activation=torch.nn.Hardswish)
    #
    # #y = model(x)
    # y = model.forward(x)
    x = torch.randn(1, 3, 8, 224, 224)
    model = efficientformer_l0(pretrained_2d=r"C:\Users\andreyan\Downloads\efficientformer_l1_1000d.pth")

    y = model(x)

    print(x.shape, y.shape)

    with torch.no_grad():
      macs, params = get_model_complexity_info(model, (3, 8, 224, 224), as_strings=True,
                                              print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    # output_file = "efficientformer3d.onnx"
    # torch.onnx.export(
    #     model,
    #     args=(x),
    #     input_names=["x"],
    #     output_names=["out"],
    #     f=output_file,
    #     export_params=True,
    #     keep_initializers_as_inputs=True,
    #     verbose=True,
    #     opset_version=11)