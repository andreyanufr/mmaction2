# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import AvgConsensus, BaseHead


@HEADS.register_module()
class EVLHead(BaseHead):
    """Class head for EVL.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss').
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        consensus (dict): Consensus config dict.
        dropout_ratio (float): Probability of dropout layer. Default: 0.4.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='CrossEntropyLoss'),
                 dropout_ratio=0.5,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)

        self.dropout_ratio = dropout_ratio

        self.proj = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Dropout(dropout_ratio),
            nn.Linear(in_channels, num_classes),
        )

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.proj, std=0.001)

    def forward(self, x, num_segs):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.
            num_segs (int): Number of segments into which a video
                is divided.
        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels]
        x = x.view(x.shape[0], x.shape[-1])
        cls_score = self.proj(x)
        # [N, num_classes]
        return cls_score
