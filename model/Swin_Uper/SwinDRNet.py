# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

from .SwinTransformer import SwinTransformerSys
from .UPerNet import UPerHead
from .CrossAttention import CrossAttention

logger = logging.getLogger(__name__)


class SwinDRNet(nn.Module):
    """ SwinDRNet.
        A PyTorch impl of SwinDRNet, a depth restoration network proposed in: 
        `Domain Randomization-Enhanced Depth Simulation and Restoration for 
        Perceiving and Grasping Specular and Transparent Objects' (ECCV2022)
    """

    def __init__(self, config, img_size=224, num_classes=3, logger=None):
        super(SwinDRNet, self).__init__()
        self.num_classes = num_classes
        self.config = config
        self.img_size = img_size

        self.backbone_xyz_branch = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                                      patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                      in_chans=config.MODEL.SWIN.IN_CHANS,
                                                      embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                                      depths=config.MODEL.SWIN.DEPTHS,
                                                      num_heads=config.MODEL.SWIN.NUM_HEADS,
                                                      window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                                      mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                                      qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                                      qk_scale=config.MODEL.SWIN.QK_SCALE,
                                                      drop_rate=config.MODEL.DROP_RATE,
                                                      drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                                      ape=config.MODEL.SWIN.APE,
                                                      patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                                      use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                                      logger=logger)

        self.decode_head_depth_restoration = UPerHead(num_classes=3, in_channels=[96, 192, 384, 768],
                                                      img_size=self.img_size)
        self.decode_head_confidence = UPerHead(num_classes=3, in_channels=[96, 192, 384, 768],
                                               img_size=self.img_size)
        self.conv1 = nn.Conv2d(3, 1, kernel_size=(1, 1), stride=(1, 1))
        self.liner1 = nn.Linear(224 * 224, 224)
        self.liner2 = nn.Linear(224, 3)

    def forward(self, depth):
        """Forward function."""

        depth = depth.repeat(1, 3, 1, 1) if depth.size()[1] == 1 else depth  # B, C, H, W

        input_org_shape = depth.shape[2:]
        depth_feature = self.backbone_xyz_branch(depth)

        output_map = self.decode_head_depth_restoration(depth_feature, input_org_shape)
        output_max_map = self.decode_head_confidence(depth_feature, input_org_shape)
        out_x = output_map[:, 0, :, :].unsqueeze(1)
        out_y = output_map[:, 1, :, :].unsqueeze(1)
        out_z = output_map[:, 2, :, :].unsqueeze(1)
        out_x_max = output_max_map[:, 0, :, :].unsqueeze(1)
        out_y_max = output_max_map[:, 1, :, :].unsqueeze(1)
        out_z_max = output_max_map[:, 2, :, :].unsqueeze(1)

        return (out_x, out_y, out_z), (out_x_max, out_y_max, out_z_max)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        self.backbone_rgb_branch.init_weights(pretrained=pretrained)
        self.backbone_xyz_branch.init_weights(pretrained=pretrained)
        self.decode_head_confidence.init_weights()
        self.decode_head_depth_restoration.init_weights()
        self.cross_attention_0.init_weights()
        self.cross_attention_1.init_weights()
        self.cross_attention_2.init_weights()
        self.cross_attention_3.init_weights()


if __name__ == '__main__':
    net = SwinDRNet()
