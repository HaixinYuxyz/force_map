import json, os
import numpy as np
import torch, cv2
from PIL import Image, ImageDraw
from torch.utils.data import Dataset
from glob import glob
import torchvision.transforms as T
from imgaug import augmenters as iaa
import imgaug as ia
from torchvision import transforms
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def get_txt_data(path):
    force_list = np.array([], dtype=np.float32)
    f = open(path)
    line = True
    while line:
        line = f.readline()
        num = line[1:-2].split(',')
        if len(num) != 3:
            break
        force_list = np.append(force_list, np.array([float(num[0]), float(num[1]), float(num[2])], dtype=np.float32))

    f.close()
    return force_list.reshape(-1, 3)


def read_mask(label_path):
    label_data = json.load(open(label_path))
    segs = label_data["shapes"][0]["points"]
    img_shape = [label_data["imageHeight"], label_data["imageWidth"]]
    mask = np.zeros(img_shape, dtype=np.uint8)
    mask_image = Image.fromarray(mask)
    xy = list(map(tuple, segs))
    ImageDraw.Draw(mask_image).polygon(xy=xy, outline=1, fill=255)
    mask_image = np.array(mask_image)
    mask = mask_image.astype(np.float32)
    return mask


# seq = iaa.Sequential([
#     iaa.Resize({
#         "height": 256,
#         "width": 256
#     }, interpolation='nearest'),
# ])

seq = iaa.Sequential([
    iaa.Resize({
        "height": 256,
        "width": 256
    }, interpolation='nearest'),
    iaa.Affine(rotate=(-60, 60)),
    iaa.SomeOf(1, [
        iaa.Fliplr(p=1),  # 水平翻转
        iaa.Flipud(p=1),  # 垂直翻转
        # iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊
        # iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),  # 锐化处理
        # iaa.CropAndPad(px=(-10, 0), percent=None, pad_mode='constant', pad_cval=0, keep_size=True),  # 裁剪缩放
        # iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  # 对比度增强，0.75-1.5随机数值为alpha，该alpha应用于每个通道
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # 高斯噪声
        iaa.Multiply((0.8, 1.2), per_channel=0.2), ])
], random_order=True)


# iaa.SomeOf(1, [
# iaa.Fliplr(p=1),  # 水平翻转
# iaa.Flipud(p=1),  # 垂直翻转
# iaa.Affine(rotate=(-40, 40)),
# iaa.GaussianBlur(sigma=(0, 3.0)),  # 高斯模糊
# iaa.Sharpen(alpha=(0, 0.3), lightness=(0.9, 1.1)),  # 锐化处理
# iaa.CropAndPad(px=(-10, 0), percent=None, pad_mode='constant', pad_cval=0, keep_size=True),  # 裁剪缩放
# iaa.ContrastNormalization((0.75, 1.5), per_channel=True),  # 对比度增强，0.75-1.5随机数值为alpha，该alpha应用于每个通道
# iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # 高斯噪声
# iaa.Multiply((0.8, 1.2), per_channel=0.2),])  # 20%的图片像素值乘以0.8-1.2中间的数值,用以增加图片明亮度或改变颜色
# ])])


# augs_train = iaa.Sequential([
#     # Geometric Augs
#     iaa.Resize({
#         "height": 256,
#         "width": 256
#     }, interpolation='nearest'),
#
#     # iaa.Resize({
#     #     "height": 224,
#     #     "width": 224
#     # }, interpolation='nearest'),
#     # iaa.Fliplr(0.5),
#     # iaa.Flipud(0.5),
#     # iaa.Rot90((0, 4)),
#
#     # Bright Patches
#     iaa.Sometimes(
#         0.1,
#         iaa.blend.Alpha(factor=(0.2, 0.7),
#                         first=iaa.blend.SimplexNoiseAlpha(first=iaa.Multiply((1.5, 3.0), per_channel=False),
#                                                           upscale_method='cubic',
#                                                           iterations=(1, 2)),
#                         name="simplex-blend")),
#
#     # Color Space Mods
#     iaa.Sometimes(
#         0.3,
#         iaa.OneOf([
#             iaa.Add((20, 20), per_channel=0.7, name="add"),
#             iaa.Multiply((1.3, 1.3), per_channel=0.7, name="mul"),
#             iaa.WithColorspace(to_colorspace="HSV",
#                                from_colorspace="RGB",
#                                children=iaa.WithChannels(0, iaa.Add((-200, 200))),
#                                name="hue"),
#             iaa.WithColorspace(to_colorspace="HSV",
#                                from_colorspace="RGB",
#                                children=iaa.WithChannels(1, iaa.Add((-20, 20))),
#                                name="sat"),
#             iaa.ContrastNormalization((0.5, 1.5), per_channel=0.2, name="norm"),
#             iaa.Grayscale(alpha=(0.0, 1.0), name="gray"),
#         ])),
#
#     # Blur and Noise
#     iaa.Sometimes(
#         0.2,
#         iaa.SomeOf((1, None), [
#             iaa.OneOf([iaa.MotionBlur(k=3, name="motion-blur"),
#                        iaa.GaussianBlur(sigma=(0.5, 1.0), name="gaus-blur")]),
#             iaa.OneOf([
#                 iaa.AddElementwise((-5, 5), per_channel=0.5, name="add-element"),
#                 iaa.MultiplyElementwise((0.95, 1.05), per_channel=0.5, name="mul-element"),
#                 iaa.AdditiveGaussianNoise(scale=0.01 * 255, per_channel=0.5, name="guas-noise"),
#                 iaa.AdditiveLaplaceNoise(scale=(0, 0.01 * 255), per_channel=True, name="lap-noise"),
#                 iaa.Sometimes(1.0, iaa.Dropout(p=(0.003, 0.01), per_channel=0.5, name="dropout")),
#             ]),
#         ],
#                    random_order=True))
# ])
#
# # Validation Dataset
# augs_test = iaa.Sequential([
#     iaa.Resize({
#         "height": 256,
#         "width": 256
#     }, interpolation='nearest'),
# ])
#
# input_only = [
#     "simplex-blend", "add", "mul", "hue", "sat", "norm", "gray", "motion-blur", "gaus-blur", "add-element",
#     "mul-element", "guas-noise", "lap-noise", "dropout", "cdropout"
# ]


class ForceData(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()
        self.force = get_txt_data(os.path.join(data_path, 'data_force.txt'))  # x:-1.4656 1.1671 y: -0.9182 1.6964 z: -4.0909 -1.5137
        self.torque = get_txt_data(os.path.join(data_path, 'data_torque.txt'))
        self.rgb = glob(os.path.join(data_path, "rgb_cut", "*.jpg"))
        self.inf = glob(os.path.join(data_path, "inf_cut", "*.jpg"))
        self.lable = glob(os.path.join(data_path, "rgb_mask", "*.jpg"))
        self.rgb.sort()
        self.inf.sort()
        self.lable.sort()
        self.transform = seq

        self.scale = [1, 1, 1]

        self.transform1 = T.Compose([
            T.ToPILImage(),
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

    def totensor(self, img):
        '''
        convert numpy array to tensor
        '''
        # transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        # ])

        img = img.astype(np.float32) / 255.0
        img -= img.mean()
        img = img.transpose((2, 0, 1))

        return torch.from_numpy(img.astype(np.float32))

    def mask_transform(self, mask):
        mask = Image.fromarray(mask)
        mask = mask.resize((256, 256), Image.NEAREST)
        mask = torch.from_numpy(np.array(mask).astype(np.float32) / 255)
        return mask.unsqueeze(0)

    def _activator_masks(self, images, augmenter, parents, default):
        '''Used with imgaug to help only apply some augmentations to images and not labels
        Eg: Blur is applied to input only, not label. However, resize is applied to both.
        '''
        if self.input_only and augmenter.name in self.input_only:
            return False
        else:
            return default

    def __getitem__(self, index):

        inf = cv2.resize(cv2.imread(self.inf[index]), (224, 224))
        rgb = cv2.resize(cv2.imread(self.rgb[index]), (224, 224))
        mask = cv2.resize(cv2.imread(self.lable[index], cv2.IMREAD_GRAYSCALE), (224, 224)) / 255.

        # mask = SegmentationMapsOnImage(mask, shape=inf.shape)
        # inf, mask = self.transform(image=inf, segmentation_maps=mask)
        # mask = mask.draw(size=mask.shape[:2])[0][:, :, 0] / mask.draw(size=mask.shape[:2])[0][:, :, 0].max()
        # x:-1.4656 1.1671 y: -0.9182 1.6964 z: -4.0909 -1.5137
        x_force = mask * self.force[index][0] / self.scale[0]
        x_force = (x_force + 1.4656) / (1.1671 + 1.4656) * mask
        y_force = mask * self.force[index][1] / self.scale[1]
        y_force = (y_force + 0.9182) / (0.9182 + 1.6964) * mask
        z_force = mask * (-self.force[index][2]) / self.scale[2]
        z_force = (z_force - 1.5137) / (4.0909 - 1.5137) * mask

        inf = transforms.ToTensor()(np.ascontiguousarray(inf))
        rgb = transforms.ToTensor()(np.ascontiguousarray(rgb))
        force_map_x = transforms.ToTensor()(np.ascontiguousarray(x_force))
        force_map_y = transforms.ToTensor()(np.ascontiguousarray(y_force))
        force_map_z = transforms.ToTensor()(np.ascontiguousarray(z_force))
        mask = transforms.ToTensor()(np.ascontiguousarray(mask))

        return rgb.to(torch.float32), inf.to(torch.float32), (force_map_x.to(torch.float32), force_map_y.to(torch.float32), force_map_z.to(torch.float32)), mask.to(
            torch.float32), (self.force[index][0], self.force[index][1], self.force[index][2])

    def __len__(self):
        return len(self.inf)
