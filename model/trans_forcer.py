from .swin_transformer_v2 import swin_transformer_v2_t
from .fcn_decoder import FcnDecoder
from torch import nn
import torch
from torchvision.models import resnet18

class TransForcer(nn.Module):
    def __init__(self, in_channels, window_size, input_shape, checkout_path, use_checkout, **decoder_param):
        super().__init__()
        self.encoder = swin_transformer_v2_t(input_shape, window_size, in_channels, sequential_self_attention=False, use_checkpoint=False)
        if use_checkout:
            checkpoint = torch.load(checkout_path)
            self.encoder.load_state_dict(checkpoint)
        self.decoder = FcnDecoder(**decoder_param)
    
    def forward(self, inf):
        # x = torch.cat((rgb, inf), dim=1)
        features = self.encoder(inf)
        features.reverse()
        out = self.decoder(features)
        return out

# resnet18 = resnet18(progress=False)
# resnet18.avgpool = nn.Identity()
# resnet18.fc = nn.Identity()
# x = torch.randn(1,3,256,256)
# y = resnet18(x)
# print(y.shape)