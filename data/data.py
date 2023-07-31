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
import numpy.matlib
import pandas
import math


def read_tablemethod(filename):
    data = pandas.read_table(filename, header=None, delim_whitespace=True)
    return data


def get_radius_model(data_path):
    radius_model_dict = {}
    for radius in [0.5, 1, 1.5, 2, 2.5, 3]:
        temp_dict = {}
        for axis in ['x', 'y', 'z']:
            force_xyz = axis
            data = read_tablemethod(os.path.join(data_path, 'new_map', '{}.txt'.format(radius)))
            if force_xyz == "x" or force_xyz == "y":
                idx = 2
            else:
                idx = 3
            res = []
            k = len(data[idx].values)
            for i in range(0, k):
                if float(data[idx].values[i]) != 0:
                    num = round(data[idx].values[i] / 1000, 4)
                    res.append(num)
            m = len(res)
            fit_id = np.arange(-(m - 1) / 2, (m + 1) / 2)
            z1 = np.polyfit(fit_id, res, 10)
            p1 = np.poly1d(z1)
            model_img = np.zeros((m, m))
            for i in range(0, m):
                for j in range(0, m):
                    dis = math.sqrt(abs((i - m / 2) * (i - m / 2)) + abs((j - m / 2) * (j - m / 2)))
                    if dis > (m - 1) / 2:
                        model_img[i][j] = 0
                    else:
                        model_img[i][j] = abs(p1(dis))
            temp_dict.update({axis: model_img})
        radius_model_dict.update({str(radius): temp_dict})
    return radius_model_dict


def picture_trans(force_xyz, mask, model_img_x, model_img_y, model_img_z):
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 200:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if w > h:
            h = w
        else:
            w = h
    resized_x = cv2.resize(model_img_x, (w, w), interpolation=cv2.INTER_AREA)
    resized_y = cv2.resize(model_img_y, (w, w), interpolation=cv2.INTER_AREA)
    resized_z = cv2.resize(model_img_z, (w, w), interpolation=cv2.INTER_AREA)

    force_x = np.zeros_like(mask)
    force_x[y:y + w, x:x + w] = resized_x
    force_x = force_x / np.max(model_img_x) * force_xyz[0]

    force_y = np.zeros_like(mask)
    force_y[y:y + w, x:x + w] = resized_y
    force_y = force_y / np.max(model_img_y) * force_xyz[1]

    force_z = np.zeros_like(mask)
    force_z[y:y + w, x:x + w] = resized_z
    force_z = force_z / np.max(model_img_z) * force_xyz[2]

    return force_x, force_y, force_z


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
    def __init__(self, data_path, dataset_type) -> None:
        super().__init__()
        self.dataset_type = dataset_type
        if dataset_type == 'old':
            self.force = get_txt_data(os.path.join(data_path, 'data_force.txt'))
            self.torque = get_txt_data(os.path.join(data_path, 'data_torque.txt'))
            self.rgb = glob(os.path.join(data_path, "rgb_cut", "*.jpg"))
            self.inf = glob(os.path.join(data_path, "inf_cut", "*.jpg"))
            self.lable = glob(os.path.join(data_path, "rgb_mask", "*.jpg"))
            self.rgb.sort()
            self.inf.sort()
            self.lable.sort()
        elif dataset_type == 'new':
            self.force = np.array([]).reshape(-1, 3)
            self.torque = np.array([]).reshape(-1, 3)
            self.rgb = []
            self.inf = []
            self.lable = []
            for file_folder in glob(os.path.join(data_path, '*')):
                if os.path.split(file_folder)[1] != 'new_map' and os.path.split(file_folder)[1] != 'force_map':
                    self.force = np.append(self.force, get_txt_data(os.path.join(file_folder, 'data_force.txt')), axis=0)
                    self.torque = np.append(self.torque, get_txt_data(os.path.join(file_folder, 'data_torque.txt')), axis=0)
                    rgb = glob(os.path.join(file_folder, "rgb_cut", "*.jpg"))
                    rgb.sort()
                    self.rgb.extend(rgb)
                    inf = glob(os.path.join(file_folder, "inf_cut", "*.jpg"))
                    inf.sort()
                    self.inf.extend(inf)
                    label = glob(os.path.join(file_folder, "mask", "*.jpg"))
                    label.sort()
                    self.lable.extend(label)
        self.transform = seq
        self.force_max_x = np.max(self.force[:, 0])
        self.force_min_x = np.min(self.force[:, 0])
        self.force_max_y = np.max(self.force[:, 1])
        self.force_min_y = np.min(self.force[:, 1])
        self.force_max_z = np.max(self.force[:, 2])
        self.force_min_z = np.min(self.force[:, 2])
        self.scale = [1, 1, 1]

        self.radius_model = get_radius_model(data_path)

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

    def gauss_transformation(self, pos_img, center):
        center_x, center_y = center[0], center[1]
        IMAGE_HEIGHT = pos_img.shape[0]
        IMAGE_WIDTH = pos_img.shape[1]
        R = 0.5 * 50
        mask_x = np.matlib.repmat(center_x, IMAGE_HEIGHT, IMAGE_WIDTH)
        mask_y = np.matlib.repmat(center_y, IMAGE_HEIGHT, IMAGE_WIDTH)

        x1 = np.arange(IMAGE_WIDTH)
        x_map = np.matlib.repmat(x1, IMAGE_HEIGHT, 1)

        y1 = np.arange(IMAGE_HEIGHT)
        y_map = np.matlib.repmat(y1, IMAGE_WIDTH, 1)
        y_map = np.transpose(y_map)

        Gauss_map = np.sqrt((x_map - mask_x) ** 2 + (y_map - mask_y) ** 2)

        Gauss_map = np.exp(-0.5 * Gauss_map / R)
        guess_img = pos_img * Gauss_map
        return guess_img

    def find_max_region(self, mask_sel):
        contours, hierarchy = cv2.findContours(mask_sel, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # 找到最大区域并填充
        area = []

        for j in range(len(contours)):
            area.append(cv2.contourArea(contours[j]))

        max_idx = np.argmax(area)

        max_area = cv2.contourArea(contours[max_idx])

        for k in range(len(contours)):

            if k != max_idx:
                cv2.fillPoly(mask_sel, [contours[k]], 0)
        return mask_sel

    def get_mask_center(self, cnt):
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            M["m00"] = 1
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        return (cX, cY)

    def __getitem__(self, index):

        # 处理mask
        mask = cv2.imread(self.lable[index], cv2.IMREAD_GRAYSCALE)
        thresh, img = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        img = self.find_max_region(img)
        cnt = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        # center = self.get_mask_center(cnt[0])
        mask = mask / 255.
        path_split = self.inf[index].split('/')[2][4:]
        x_force, y_force, z_force = picture_trans(self.force[index], mask, self.radius_model[path_split]['x'], self.radius_model[path_split]['y'],
                                                  self.radius_model[path_split]['z'])

        inf = cv2.resize(cv2.imread(self.inf[index]), (224, 224))
        rgb = cv2.resize(cv2.imread(self.rgb[index]), (224, 224))
        x_force = cv2.resize(x_force, (224, 224))
        y_force = cv2.resize(y_force, (224, 224))
        z_force = cv2.resize(z_force, (224, 224))
        mask = cv2.resize(mask, (224, 224))

        # mask_gauss = self.gauss_transformation(mask, center)

        # x_force = mask * self.force[index][0]
        x_force = (x_force - self.force_min_x) / (self.force_max_x - self.force_min_x)
        # y_force = mask * self.force[index][1]
        y_force = (y_force - self.force_min_y) / (self.force_max_y - self.force_min_y)
        # z_force = mask * (self.force[index][2])
        z_force = (z_force - self.force_min_z) / (self.force_max_z - self.force_min_z)

        inf = transforms.ToTensor()(np.ascontiguousarray(inf))
        rgb = transforms.ToTensor()(np.ascontiguousarray(rgb))
        force_map_x = transforms.ToTensor()(np.ascontiguousarray(x_force))
        force_map_y = transforms.ToTensor()(np.ascontiguousarray(y_force))
        force_map_z = transforms.ToTensor()(np.ascontiguousarray(z_force))
        mask = transforms.ToTensor()(np.ascontiguousarray(mask))
        # mask_gauss = transforms.ToTensor()(np.ascontiguousarray(mask_gauss))

        return rgb.to(torch.float32), inf.to(torch.float32), (force_map_x.to(torch.float32), force_map_y.to(torch.float32), force_map_z.to(torch.float32)), mask.to(
            torch.float32), (self.force[index][0], self.force[index][1], self.force[index][2]), self.dataset_type, [self.force_max_x,
                                                                                                                    self.force_min_x,
                                                                                                                    self.force_max_y,
                                                                                                                    self.force_min_y,
                                                                                                                    self.force_max_z,
                                                                                                                    self.force_min_z], \
               torch.tensor([torch.max(force_map_x), torch.max(force_map_y), torch.max(force_map_y)])

    def __len__(self):
        return len(self.inf)
