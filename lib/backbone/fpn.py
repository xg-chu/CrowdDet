# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import math

import torch
from torch import nn
import torch.nn.functional as F

from layers.conv2d import Conv2d

class FPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up):
        super(FPN, self).__init__()
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        use_bias =True

        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = Conv2d(
                in_channels, fpn_dim, kernel_size=1, bias=use_bias)
            output_conv = Conv2d(
                fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1, bias=use_bias)
            nn.init.kaiming_uniform_(lateral_conv.weight, a=1)
            nn.init.constant_(lateral_conv.bias, 0)
            nn.init.kaiming_uniform_(output_conv.weight, a=1)
            nn.init.constant_(output_conv.bias, 0)
            self.add_module("fpn_lateral{}".format(idx), lateral_conv)
            self.add_module("fpn_output{}".format(idx), output_conv)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[::-1]
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))
        for features, lateral_conv, output_conv in zip(
            bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:]
        ):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="nearest")
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))

        last_results = F.avg_pool2d(results[0], kernel_size=2, stride=2, padding=0)
        results.insert(0, last_results)
        return results

