from turtle import forward
import torch
from torchvision.models.segmentation import fcn_resnet50
from torch import nn


class resnet_fcn(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        model = fcn_resnet50()
        model.backbone.conv1 = nn.Conv2d(6, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.classifier[-1] = nn.Identity()
        self.branch = model

        self.out_x = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.out_y = nn.Conv2d(512, 1, kernel_size=1, stride=1)
        self.out_z = nn.Conv2d(512, 1, kernel_size=1, stride=1)

    def forward(self, rgb, inf):
        x = torch.cat((rgb, inf), dim=1)
        x = self.branch(x)["out"]

        out_x = self.out_x(x)
        out_y = self.out_y(x)
        out_z = self.out_z(x)

        return (out_x, out_y, out_z)


if __name__ == "__main__":
    model = resnet_fcn()
    rgb = torch.randn(1, 3, 256, 256)
    inf = torch.randn(1, 3, 256, 256)
    x, y, z = model(rgb, inf)
    print(x.shape)
