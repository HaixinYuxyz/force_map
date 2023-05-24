# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from .swin_transformer_unet_skip_expand_decoder_sys import SwinTransformerSys_RGB, SwinTransformerSys_inf, SwinTransformerSys_UP
from .CrossAttention import CrossAttention

logger = logging.getLogger(__name__)


class SwinUnet_nocross(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False):
        super(SwinUnet_nocross, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.config = config

        self.swin_unet_rgb = SwinTransformerSys_RGB(img_size=config.DATA.IMG_SIZE,
                                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                                    num_classes=self.num_classes,
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
                                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.swin_unet_inf = SwinTransformerSys_inf(img_size=config.DATA.IMG_SIZE,
                                                    patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                    in_chans=config.MODEL.SWIN.IN_CHANS,
                                                    num_classes=self.num_classes,
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
                                                    use_checkpoint=config.TRAIN.USE_CHECKPOINT)
        self.swin_unet_up = SwinTransformerSys_UP(img_size=config.DATA.IMG_SIZE,
                                                  patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                                  in_chans=config.MODEL.SWIN.IN_CHANS,
                                                  num_classes=self.num_classes,
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
                                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT)

        self.cross_attention_0 = CrossAttention(in_channel=96, depth=1, num_heads=1)
        self.cross_attention_1 = CrossAttention(in_channel=192, depth=1, num_heads=1)
        self.cross_attention_2 = CrossAttention(in_channel=384, depth=1, num_heads=1)

        self.down_sample_0 = nn.Linear(in_features=96 * 3, out_features=96)
        self.down_sample_1 = nn.Linear(in_features=192 * 3, out_features=192)
        self.down_sample_2 = nn.Linear(in_features=384 * 3, out_features=384)

    def forward(self, rgb, inf):

        x_rgb, x_downsample_rgb = self.swin_unet_rgb(rgb)
        x_inf, x_downsample_inf = self.swin_unet_inf(inf)

        # x = []
        # out = self.cross_attention_0(tuple([x_downsample_rgb[0], x_downsample_inf[0]]))
        # x.append(self.down_sample_0(torch.cat((out, x_downsample_rgb[0], x_downsample_inf[0]), -1)))
        # out = self.cross_attention_1(tuple([x_downsample_rgb[1], x_downsample_inf[1]]))
        # x.append(self.down_sample_1(torch.cat((out, x_downsample_rgb[1], x_downsample_inf[1]), -1)))
        # out = self.cross_attention_2(tuple([x_downsample_rgb[2], x_downsample_inf[2]]))
        # x.append(self.down_sample_2(torch.cat((out, x_downsample_rgb[2], x_downsample_inf[2]), -1)))

        logits = self.swin_unet_up(x_rgb, x_inf)
        out_x = logits[:, 0, :, :].unsqueeze(1)
        out_y = logits[:, 1, :, :].unsqueeze(1)
        out_z = logits[:, 2, :, :].unsqueeze(1)

        return (out_x, out_y, out_z)

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.swin_unet.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3 - int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k: v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
                        del full_dict[k]

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
