import math

import torch
from torch import nn
import torch.nn.functional as F

class FPN(nn.Module):
    """
    This module implements Feature Pyramid Network.
    It creates pyramid features built on top of some input feature maps.
    """
    def __init__(self, bottom_up, layers_begin, layers_end):
        super(FPN, self).__init__()
        assert layers_begin > 1 and layers_begin < 6
        assert layers_end > 4 and layers_begin < 8
        in_channels = [256, 512, 1024, 2048]
        fpn_dim = 256
        in_channels = in_channels[layers_begin-2:]

        lateral_convs = nn.ModuleList()
        output_convs = nn.ModuleList()
        for idx, in_channels in enumerate(in_channels):
            lateral_conv = nn.Conv2d(in_channels, fpn_dim, kernel_size=1)
            output_conv = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=1, padding=1)
            nn.init.kaiming_normal_(lateral_conv.weight, mode='fan_out')
            nn.init.constant_(lateral_conv.bias, 0)
            nn.init.kaiming_normal_(output_conv.weight, mode='fan_out')
            nn.init.constant_(output_conv.bias, 0)
            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)

        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]
        self.bottom_up = bottom_up
        self.output_b = layers_begin
        self.output_e = layers_end
        if self.output_e == 7:
            self.p6 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
            self.p7 = nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, stride=2, padding=1)
            for l in [self.p6, self.p7]:
                nn.init.kaiming_uniform_(l.weight, a=1)  # pyre-ignore
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        bottom_up_features = self.bottom_up(x)
        bottom_up_features = bottom_up_features[self.output_b - 2:]
        bottom_up_features = bottom_up_features[::-1]
        results = []
        prev_features = self.lateral_convs[0](bottom_up_features[0])
        results.append(self.output_convs[0](prev_features))
        for l_id, (features, lateral_conv, output_conv) in enumerate(zip(
            bottom_up_features[1:], self.lateral_convs[1:], self.output_convs[1:])):
            top_down_features = F.interpolate(prev_features, scale_factor=2, mode="bilinear", align_corners=False)
            lateral_features = lateral_conv(features)
            prev_features = lateral_features + top_down_features
            results.append(output_conv(prev_features))
        if(self.output_e == 6):
            p6 = F.max_pool2d(results[0], kernel_size=1, stride=2, padding=0)
            results.insert(0, p6)
        elif(self.output_e == 7):
            p6 = self.p6(results[0])
            results.insert(0, p6)
            p7 = self.p7(F.relu(results[0]))
            results.insert(0, p7)
        return results

