# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch.nn as nn
import torch.nn.functional as F


class Interpolate(nn.Module):  # TODO: Not sure about normalization generally (norm_cfg in original code)
    def __init__(self, scale_factor, mode, align_corners=False):
        super().__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)
        return x


class ReassembleBlocks(nn.Module):
    """ViTPostProcessBlock, process cls_token in ViT backbone output and
    rearrange the feature vector to feature map.
    Args:
        in_channels (int): ViT feature channels. Default: 768.
        out_channels (List): output channels of each stage.
            Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
    """
    def __init__(
        self, in_channels=768, out_channels=None, readout_type="ignore", patch_size=16
    ):
        super().__init__()

        if out_channels is None:
            out_channels = [96, 192, 384, 384]

        assert readout_type in ["ignore"]  # ["ignore", "add", "project"]
        self.readout_type = readout_type
        self.patch_size = patch_size

        self.projects = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channel, kernel_size=1) for out_channel in out_channels
            ]
        )

        self.resize_layers = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                ),
                nn.ConvTranspose2d(
                    in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                ),
                nn.Identity(),
                nn.Conv2d(
                    in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                )
            ]
        )

    def forward(self, inputs):
        out = []
        for i, x in enumerate(inputs):
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)
        return out


class PreActResidualConvUnit(nn.Module):
    """ResidualConvUnit, pre-activate residual unit.
    Args:
        in_channels (int): number of channels in the input feature map.
        stride (int): stride of the first block. Default: 1
        dilation (int): dilation rate for convs layers. Default: 1.
    """
    def __init__(self, in_channels, stride=1, dilation=1, bn=False):
        super().__init__()

        self.bn = bn
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=not self.bn,
        )
        self.conv2 = nn.Conv2d(
            in_channels,
            in_channels,
            3,
            padding=1,
            bias=not self.bn,
        )
        if self.bn:
            self.bn1 = nn.BatchNorm2d(in_channels)
            self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, inputs):
        inputs_ = inputs.clone()
        x = self.act(inputs)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        
        x = self.act(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        
        return x + inputs_


class FeatureFusionBlock(nn.Module):
    """FeatureFusionBlock, merge feature map from different stages.
    Args:
        in_channels (int): Input channels.
        expand (bool): Whether expand the channels in post process block.
            Default: False.
        align_corners (bool): align_corner setting for bilinear upsample.
            Default: True.
    """
    def __init__(self, in_channels, expand=False, align_corners=True):
        super().__init__()

        self.in_channels = in_channels
        self.expand = expand
        self.align_corners = align_corners

        self.out_channels = in_channels
        if self.expand:
            self.out_channels = in_channels // 2

        self.project = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1)

        self.res_conv_unit1 = PreActResidualConvUnit(in_channels=self.in_channels)
        self.res_conv_unit2 = PreActResidualConvUnit(in_channels=self.in_channels)

    def forward(self, *inputs):
        x = inputs[0]
        if len(inputs) == 2:
            if x.shape != inputs[1].shape:
                res = F.interpolate(inputs[1], size=(x.shape[2], x.shape[3]), mode="bilinear", align_corners=False)
            else:
                res = inputs[1]
            x = x + self.res_conv_unit1(res)
        x = self.res_conv_unit2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=self.align_corners)
        x = self.project(x)
        return x


class OutputHead(nn.Module):
    def __init__(self, latent_size=768):
        super().__init__()
        # TODO: Not sure about structure
        self.head_modules = nn.ModuleList(
            [
                nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(in_channels=latent_size, out_channels=latent_size, kernel_size=2, stride=2, padding=0),
                nn.Conv2d(in_channels=latent_size, out_channels=latent_size, kernel_size=3, stride=1, padding=1),
            ]
        )

    def forward(self, x):
        for module in self.head_modules:
            x = module(x)
        return x


class DPTHead(nn.Module):
    """Vision Transformers for Dense Prediction.
    This head is implemented of `DPT <https://arxiv.org/abs/2103.13413>`_.
    Args:
        embed_dims (int): The embed dimension of the ViT backbone.
            Default: 768.
        post_process_channels (List): Out channels of post process conv
            layers. Default: [96, 192, 384, 768].
        readout_type (str): Type of readout operation. Default: 'ignore'.
        patch_size (int): The patch size. Default: 16.
        expand_channels (bool): Whether expand the channels in post process
            block. Default: False.
    """
    def __init__(
        self,
        embed_dims=768,
        post_process_channels=None,
        readout_type="ignore",
        patch_size=16,
        d_out=384,
        expand_channels=False,
    ):
        super().__init__()

        if not post_process_channels:
            self.post_process_channels = [96, 192, 384, 768]
        self.post_process_channels = [min(d_out, channel) for channel in post_process_channels]

        self.d_out = d_out
        self.expand_channels = expand_channels
        self.reassemble_blocks = ReassembleBlocks(embed_dims, self.post_process_channels, readout_type, patch_size)
        self.convs = nn.ModuleList()
        for channel in self.post_process_channels:
            self.convs.append(nn.Conv2d(channel, self.d_out, kernel_size=3, padding=1, bias=False))
        self.fusion_blocks = nn.ModuleList()
        for _ in range(len(self.convs)):
            self.fusion_blocks.append(FeatureFusionBlock(self.d_out))
        self.fusion_blocks[0].res_conv_unit1 = None
        self.project = nn.Conv2d(self.d_out, self.d_out, kernel_size=3, padding=1)
        self.output_head = OutputHead(d_out)

        self.num_fusion_blocks = len(self.fusion_blocks)
        self.num_reassemble_blocks = len(self.reassemble_blocks.resize_layers)
        self.num_post_process_channels = len(self.post_process_channels)
        assert self.num_fusion_blocks == self.num_reassemble_blocks
        assert self.num_reassemble_blocks == self.num_post_process_channels

    def forward(self, inputs):
        assert len(inputs) == self.num_reassemble_blocks
        x = [inp for inp in inputs]
        x = self.reassemble_blocks(x)
        x = [self.convs[i](feature) for i, feature in enumerate(x)]
        out = self.fusion_blocks[0](x[-1])
        for i in range(1, len(self.fusion_blocks)):
            out = self.fusion_blocks[i](out, x[-(i + 1)])
        out = self.project(out)
        out = self.output_head(out)
        return [out]  # list for BTS
