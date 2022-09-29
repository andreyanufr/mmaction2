# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead
from timm.models.vision_transformer import trunc_normal_

class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        #self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('bn', torch.nn.LayerNorm(a))
        l = torch.nn.Linear(a, b, bias=bias)
        trunc_normal_(l.weight, std=std)
        if bias:
            torch.nn.init.constant_(l.bias, 0)
        self.add_module('l', l)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


@HEADS.register_module()
class LevitHead(BaseHead):
    """Classification head for GCViViT.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        fc1_bias (bool): If the first fc layer has bias. Default: False.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 norm_layer=nn.LayerNorm):
        super().__init__(num_classes, in_channels, loss_cls)
        self.head = BN_Linear(in_channels, num_classes) 
        self.apply(self._init_weights)

    def init_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.clip(m.weight, min=-.02, max=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [B, T, IN_CHANNELS]
        #B, T, C = x.shape
        #x = x.view(B * T, -1)
        #print('SHAPE head', x.shape)
        cls_score = self.head(x)
        #print('SHAPE cls score', x.shape)
        if len(cls_score.shape) > 2:
            cls_score = torch.mean(cls_score, dim=1)
        #print("Shape 1: ", cls_score.shape)
        #cls_score = torch.mean(cls_score, dim=1)
        #print("Shape 2 mean: ", cls_score.shape)
        # [N, num_classes]
        return cls_score
