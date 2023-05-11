from torch import nn
from mmcv.cnn import ConvModule
import torch
from mmseg.models.utils import Upsample
from timm.models.layers.cbam import *


class FcnDecoder(nn.Module):
    def __init__(self, depth, hidden_dim, norm_type, act_type):
        super(FcnDecoder, self).__init__()
        self.hidden_dim = [hidden_dim * (2 ** i) for i in range(depth - 1, -1, -1)]
        self.drop = nn.Dropout(0.2)
        self.up_shape = nn.ModuleList()
        self.dw_block = nn.ModuleList()
        for i in range(depth - 1):
            self.up_shape.append(
                Upsample(
                    scale_factor=2,
                    mode='bilinear',
                    align_corners=False)
            )
            self.dw_block.append(
                nn.Sequential(
                    CbamModule(channels=self.hidden_dim[i] + self.hidden_dim[i + 1]),
                    ConvModule(
                        in_channels=self.hidden_dim[i] + self.hidden_dim[i + 1],
                        out_channels=self.hidden_dim[i + 1],
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        norm_cfg=norm_type,
                        act_cfg=act_type
                    ),
                    ConvModule(
                        in_channels=self.hidden_dim[i + 1],
                        out_channels=self.hidden_dim[i + 1],
                        kernel_size=5,
                        stride=1,
                        padding=2,
                        norm_cfg=norm_type,
                        act_cfg=act_type
                    )
                )
            )

        self.out_cache = nn.Sequential(
            Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False),
            ConvModule(
                in_channels=self.hidden_dim[-1],
                out_channels=self.hidden_dim[-1],
                kernel_size=5,
                stride=1,
                padding=2,
                norm_cfg=norm_type,
                act_cfg=act_type),
            ConvModule(
                in_channels=self.hidden_dim[-1],
                out_channels=self.hidden_dim[-1],
                kernel_size=5,
                stride=1,
                padding=2,
                norm_cfg=norm_type,
                act_cfg=act_type
            ),
        )

        self.out_x = nn.Sequential(
            nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=3, padding=1),
            Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False),
        )
        self.out_y = nn.Sequential(
            nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=3, padding=1),
            Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False),
        )
        self.out_z = nn.Sequential(
            nn.Conv2d(self.hidden_dim[-1], 1, kernel_size=3, padding=1),
            Upsample(
                scale_factor=2,
                mode='bilinear',
                align_corners=False),
        )

    def forward(self, features):
        for i in range(0, len(features) - 1):
            x = self.up_shape[i](features[i])
            x = torch.cat((x, features[i + 1]), dim=1)
            x = self.dw_block[i](x)
        x = self.drop(x)
        out = self.out_cache(x)

        out_x = self.out_x(out)
        out_y = self.out_y(out)
        out_z = self.out_z(out)
        return (out_x, out_y, out_z)
