from model.swin_transformer_v2 import swin_transformer_v2_t
from model.swin_transformer_v2.fcn_decoder import FcnDecoder
from torch import nn
import torch


class TransForcer(nn.Module):
    def __init__(self, in_channels, window_size, input_shape, checkout_path, use_checkout, logger, **decoder_param):
        super().__init__()
        self.encoder = swin_transformer_v2_t(input_shape, window_size, in_channels, sequential_self_attention=False,
                                             use_checkpoint=False)
        if use_checkout:
            checkpoint = torch.load(checkout_path)
            model_dict = self.encoder.state_dict()
            state_dict = {k: v for k, v in checkpoint.items() if k in model_dict.keys()}
            filtered_state_dict = {k: v for k, v in state_dict.items() if
                                   k in model_dict and state_dict[k].size() == model_dict[k].size()}
            model_dict.update(filtered_state_dict)
            self.encoder.load_state_dict(model_dict)
            # self.encoder.load_state_dict(checkpoint)
            logger.warning("load checkpoint {}".format(checkout_path))
        self.decoder = FcnDecoder(**decoder_param)

    def forward(self, rgb, inf):
        # x = torch.cat((rgb, inf), dim=1)
        x = torch.cat((rgb, inf), dim=1)
        features = self.encoder(x)
        features.reverse()
        out = self.decoder(features)
        return out

# resnet18 = resnet18(progress=False)
# resnet18.avgpool = nn.Identity()
# resnet18.fc = nn.Identity()
# x = torch.randn(1,3,256,256)
# y = resnet18(x)
# print(y.shape)
