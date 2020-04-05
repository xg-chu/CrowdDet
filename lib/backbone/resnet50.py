import torch
from torch import nn
import torch.nn.functional as F

from layers.batch_norm import FrozenBatchNorm2d
from layers.conv2d import Conv2d

has_bias = True

class BaseStem(nn.Module):
    def __init__(self):
        super(BaseStem, self).__init__()
        self.conv1 = Conv2d(
            3, 64, kernel_size=7, stride=2, padding=3, bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(64)

        for l in [self.conv1, ]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            if has_bias:
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        # x = F.relu_(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
        return x


class Bottleneck(nn.Module):
    def __init__(
            self, in_channels, bottleneck_channels, out_channels,
            stride, dilation):
        super(Bottleneck, self).__init__()

        self.downsample = None
        if in_channels != out_channels:
            down_stride = stride if dilation == 1 else 1
            self.downsample = nn.Sequential(
                Conv2d(in_channels, out_channels,
                    kernel_size=1, stride=down_stride, bias=has_bias),
                FrozenBatchNorm2d(out_channels), )
            for modules in [self.downsample, ]:
                for l in modules.modules():
                    if isinstance(l, Conv2d):
                        nn.init.kaiming_uniform_(l.weight, a=1)
                        if has_bias:
                            nn.init.constant_(l.bias, 0)
        if dilation > 1:
            stride = 1

        # The original MSRA ResNet models have stride in the first 1x1 conv
        # The subsequent fb.torch.resnet and Caffe2 ResNe[X]t implementations
        # have stride in the 3x3 conv
        self.conv1 = Conv2d(
            in_channels, bottleneck_channels, kernel_size=1, stride=stride,
            bias=has_bias)
        self.bn1 = FrozenBatchNorm2d(bottleneck_channels)

        self.conv2 = Conv2d(
            bottleneck_channels, bottleneck_channels,
            kernel_size=3, stride=1, padding=dilation,
            bias=has_bias, dilation=dilation)
        self.bn2 = FrozenBatchNorm2d(bottleneck_channels)

        self.conv3 = Conv2d(
            bottleneck_channels, out_channels, kernel_size=1, bias=has_bias)
        self.bn3 = FrozenBatchNorm2d(out_channels)

        for l in [self.conv1, self.conv2, self.conv3]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            if has_bias:
                nn.init.constant_(l.bias, 0)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        # out = F.relu_(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = F.relu_(out)
        out = F.relu(out)

        out0 = self.conv3(out)
        out = self.bn3(out0)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = F.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self, freeze_at):
        super(ResNet50, self).__init__()
        self.stem = BaseStem()

        self.stages = []
        block_counts = [3, 4, 6, 3]
        bottleneck_channels_list = [64, 128, 256, 512]
        out_channels_list = [256, 512, 1024, 2048]
        stride_list = [1, 2, 2, 2]
        in_channels = 64

        for layer_id in range(len(block_counts)):
            name = "layer" + str(layer_id)
            bottleneck_channels = bottleneck_channels_list[layer_id]
            out_channels = out_channels_list[layer_id]
            stride = stride_list[layer_id]

            blocks = []
            for _ in range(block_counts[layer_id]):
                blocks.append(
                    Bottleneck(in_channels, bottleneck_channels, out_channels,
                        stride, dilation=1))
                stride = 1
                in_channels = out_channels
            module = nn.Sequential(*blocks)
            in_channels = out_channels
            self.add_module(name, module)
            self.stages.append(name)

        self._freeze_backbone(freeze_at)

    def _freeze_backbone(self, freeze_at):
        if freeze_at < 0:
            return
        for stage_index in range(freeze_at):
            if stage_index == 0:
                m = self.stem  # stage 0 is the stem
            else:
                m = getattr(self, "layer" + str(stage_index))
            for p in m.parameters():
                p.requires_grad = False

    def forward(self, x):
        outputs = []
        x = self.stem(x)
        for stage_name in self.stages:
            x = getattr(self, stage_name)(x)
            outputs.append(x)
        return outputs
