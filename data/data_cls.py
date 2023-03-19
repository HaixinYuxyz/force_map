import json, os
import numpy as np
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as T

def read_mask(label_path):
    label_data = json.load(open(label_path))
    segs = label_data["shapes"][0]["points"]
    img_shape = [label_data["imageHeight"], label_data["imageWidth"]]
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask_image = Image.fromarray(mask)
    xy = list(map(tuple, segs))
    ImageDraw.Draw(mask_image).polygon(xy=xy, outline=1, fill=255)
    mask_image = np.array(mask_image)
    mask = mask_image.astype(np.int32) 
    return mask

class ForceData(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.force = np.loadtxt(open(data_path+"/data.csv","r"),delimiter=",")
        self.rgb = glob(os.path.join(data_path, "rgb_cut", "*.jpg"))
        self.inf = glob(os.path.join(data_path, "inf_cut", "*"))
        self.lable = glob(os.path.join(data_path, "rgb_cut", "*.json"))
        self.rgb.sort()
        self.inf.sort()
        self.lable.sort()

        self.scale = [2,2,6]

        self.transform1 = T.Compose([
            T.Resize(256),
            T.ToTensor(),
            T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
            )
        ])
        self.transform2 = T.Compose([
            T.ToPILImage(),
            T.Resize(256),
            T.ToTensor()
        ])

    def __getitem__(self, index):
        rgb = Image.open(self.rgb[index])
        inf = Image.open(self.inf[index])

        x_force = torch.tensor((self.force[index][0] + self.scale[0] / 2) / self.scale[0])
        y_force = torch.tensor((self.force[index][1] + self.scale[1] / 2) / self.scale[1])
        z_force = torch.tensor((self.force[index][2] + self.scale[2] / 2) / self.scale[2])

        rgb = self.transform1(rgb)
        inf = self.transform1(inf)

        return rgb.float(), inf.float(), (x_force.float(), y_force.float(), z_force.float())
    
    def __len__(self):
        return len(self.rgb)
