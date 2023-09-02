import os
import torch
import torch.nn.functional as F
import warnings
import datetime
import logging
import random

warnings.filterwarnings('ignore')
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from model.swin_transformer_v2.trans_forcer import TransForcer
from data.data import ForceData
from functools import reduce
import tensorboardX
import cv2
from utils.create_image_grid import create_grid_image
from model import get_network
import torch.backends.cudnn as cudnn
from utils.plot_point_fig import plot_point_fig


def log_creater(log_file_dir):
    logger = logging.getLogger(__file__)
    logger.setLevel(logging.DEBUG)

    # set two handlers
    fileHandler = logging.FileHandler(os.path.join(log_file_dir, 'log.log'), mode='w')
    fileHandler.setLevel(logging.DEBUG)
    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.DEBUG)

    # set formatter
    formatter = logging.Formatter('[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    consoleHandler.setFormatter(formatter)
    fileHandler.setFormatter(formatter)

    # add
    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)
    return logger


def guass_map_cal_max_force(output, real_force, mask, max_min_force):
    pred_x = torch.max(mask * output[0]) * (max_min_force[0] - max_min_force[1]) + max_min_force[1]
    pred_y = torch.max(mask * output[1]) * (max_min_force[2] - max_min_force[3]) + max_min_force[3]
    pred_z = torch.max(mask * output[2]) * (max_min_force[4] - max_min_force[5]) + max_min_force[5]

    real_x, real_y, real_z = real_force

    error_x = torch.abs(real_x - pred_x)
    error_y = torch.abs(real_y - pred_y)
    error_z = torch.abs(real_z - pred_z)

    return error_x, error_y, error_z


def guass_map_cal_delta(output, labels, mask, max_min_force):
    one = torch.ones_like(mask)
    zero = torch.zeros_like(mask)
    mask = torch.where(mask > 0.5, one, zero)
    mask = mask.to(torch.bool)
    pred_x, pred_y, pred_z = output[0][mask], output[1][mask], output[2][mask]
    label_x, label_y, label_z = labels[0][mask], labels[1][mask], labels[2][mask]

    thresh_x = torch.max(label_x / pred_x, pred_x / label_x)

    a1_x = ((thresh_x > 0) * (thresh_x < 1.05)).float().mean()
    a2_x = ((thresh_x > 0) * (thresh_x < 1.10)).float().mean()
    a3_x = ((thresh_x > 0) * (thresh_x < 1.25)).float().mean()

    thresh_y = torch.max(label_y / pred_y, pred_y / label_y)

    a1_y = ((thresh_y > 0) * (thresh_y < 1.05)).float().mean()
    a2_y = ((thresh_y > 0) * (thresh_y < 1.10)).float().mean()
    a3_y = ((thresh_y > 0) * (thresh_y < 1.25)).float().mean()

    thresh_z = torch.max(label_z / pred_z, pred_z / label_z)

    a1_z = ((thresh_z > 0) * (thresh_z < 1.05)).float().mean()
    a2_z = ((thresh_z > 0) * (thresh_z < 1.10)).float().mean()
    a3_z = ((thresh_z > 0) * (thresh_z < 1.25)).float().mean()

    return a1_x, a2_x, a3_x, a1_y, a2_y, a3_y, a1_z, a2_z, a3_z


def sum_mean_acc(output_map_max, max_xyz, mask, max_min_force):
    x_map, y_map, z_map = output_map_max
    pred_x = (torch.sum(mask * x_map) / torch.sum(mask))
    pred_y = (torch.sum(mask * y_map) / torch.sum(mask))
    pred_z = (torch.sum(mask * z_map) / torch.sum(mask))
    error_x = torch.abs(pred_x - torch.max(max_xyz[0]))
    error_y = torch.abs(pred_y - torch.max(max_xyz[1]))
    error_z = torch.abs(pred_z - torch.max(max_xyz[2]))
    return error_x * (max_min_force[0] - max_min_force[1]), error_y * (max_min_force[2] - max_min_force[3]), error_z * (max_min_force[4] - max_min_force[5])


def criterion(pred, pred_max, gt, gt_max):
    losses = []
    losses_max = []
    for y, yt in zip(pred, gt):
        losses.append(F.smooth_l1_loss(y, yt))
    loss_sum = reduce(lambda x, y: x + y, losses)

    for y, yt in zip(pred_max, gt_max):
        losses_max.append(F.smooth_l1_loss(y, yt))
    loss_max_sum = reduce(lambda x, y: x + y, losses_max)

    loss_all = 0.5 * loss_sum + loss_max_sum

    return losses, loss_sum, losses_max, loss_max_sum, loss_all


def train(net, criterion, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, stats_dir,
        optim='sgd', init=True, scheduler_type='Cosine'):
    def init_xavier(m):  # 参数初始化
        # if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)
    iter_per_train = len(train_dataloader)
    optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    warm_up_with_multistep_lr = lambda epoch: epoch / (20 * iter_per_train) if epoch <= (20 * iter_per_train) else 0.4 ** len(
        [m for m in [40 * iter_per_train, 60 * iter_per_train, 80 * iter_per_train, 90 * iter_per_train] if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    tb = tensorboardX.SummaryWriter(stats_dir)

    total_iter_num = 0
    error_mean_max_best = 100000
    error_guass_delta_best = 0
    for epoch in range(num_epoch):
        print("——————{} epoch——————".format(epoch + 1))

        net.train()
        for batch in tqdm(train_dataloader, desc='training'):
            rgb, inf, targets, mask, real_force, d_type, max_min_forc, max_xyz, force_maxs = batch
            inf = inf.to(device)
            labels_map = [target.to(device) for target in targets]
            labels_max = [force_max.to(device) for force_max in force_maxs]

            output_map, output_map_max = net(inf)
            losses, loss_sum, losses_max, loss_max_sum, loss_all = criterion(output_map, output_map_max, labels_map, labels_max)
            optimizer.zero_grad()
            loss_all.backward()
            optimizer.step()

            current_learning_rate = optimizer.param_groups[0]['lr']
            tb.add_scalar('Learning_Rate', current_learning_rate, total_iter_num)

            if total_iter_num % 20 == 0:
                if not os.path.exists(os.path.join(stats_dir, 'log_img')):
                    os.makedirs(os.path.join(stats_dir, 'log_img'))
                grid_image = create_grid_image(rgb=rgb, inf=inf, mask=mask, output=output_map, labels=labels_map, max_num_images_to_save=6)
                tb.add_image('Train image', grid_image, total_iter_num)
                cv2.imwrite(os.path.join(stats_dir, 'log_img', '{}_{}pic_train.png'.format(epoch, total_iter_num)),
                            grid_image.permute(1, 2, 0).cpu().detach().numpy() * 255)
            total_iter_num += 1
            scheduler.step()

        # =============================================================================================================
        net.eval()
        error_guass_max = {'error_x': [], 'error_y': [], 'error_z': []}

        error_guass_delta = {'mask_pix': [], 'a1_x': [], 'a2_x': [], 'a3_x': [], 'a1_y': [], 'a2_y': [], 'a3_y': [], 'a1_z': [], 'a2_z': [], 'a3_z': []}
        error_max_xyz = {'max_x': [], 'max_y': [], 'max_z': []}
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_dataloader), desc='valling'):
                rgb, inf, targets, mask, real_force, d_type, max_min_force, max_xyz, force_maxs = batch
                real_force = [force.to(device) for force in real_force]
                max_min_force = [max_min.to(device) for max_min in max_min_force]
                inf = inf.to(device)
                mask = mask.to(device)
                labels_map = [target.to(device) for target in targets]
                output_map, output_map_max = net(inf)

                # 以最大值的方式评估
                error_guass_max_x, error_guass_max_y, error_guass_max_z = guass_map_cal_max_force(output_map, real_force, mask, max_min_force)
                error_guass_max['error_x'].append(error_guass_max_x.item())
                error_guass_max['error_y'].append(error_guass_max_y.item())
                error_guass_max['error_z'].append(error_guass_max_z.item())

                # 以深度补全的方式进行评估
                a1_x, a2_x, a3_x, a1_y, a2_y, a3_y, a1_z, a2_z, a3_z = guass_map_cal_delta(output_map, labels_map, mask, max_min_force)
                error_guass_delta['mask_pix'].append(torch.sum(mask).item())
                error_guass_delta['a1_x'].append(a1_x.item())
                error_guass_delta['a2_x'].append(a2_x.item())
                error_guass_delta['a3_x'].append(a3_x.item())

                error_guass_delta['a1_y'].append(a1_y.item())
                error_guass_delta['a2_y'].append(a2_y.item())
                error_guass_delta['a3_y'].append(a3_y.item())

                error_guass_delta['a1_z'].append(a1_z.item())
                error_guass_delta['a2_z'].append(a2_z.item())
                error_guass_delta['a3_z'].append(a3_z.item())

                # 使用平均值的方式进行评估
                error_x, error_y, error_z = sum_mean_acc(output_map_max, max_xyz, mask, max_min_force)
                error_max_xyz['max_x'].append(error_x)
                error_max_xyz['max_y'].append(error_y)
                error_max_xyz['max_z'].append(error_z)

                if idx % 20 == 0:
                    if not os.path.exists(os.path.join(stats_dir, 'log_img')):
                        os.makedirs(os.path.join(stats_dir, 'log_img'))
                    grid_image = create_grid_image(rgb=rgb, inf=inf, mask=mask, output=output_map, labels=labels_map, max_num_images_to_save=6)
                    cv2.imwrite(os.path.join(stats_dir, 'log_img', '{}_{}pic_test.png'.format(epoch, idx)),
                                grid_image.permute(1, 2, 0).cpu().detach().numpy() * 255)

            np.save(os.path.join(stats_dir, 'error_guass_max_{}.npy'.format(epoch)), error_guass_max)
            np.save(os.path.join(stats_dir, 'error_guass_delta_{}.npy'.format(epoch)), error_guass_delta)
            np.save(os.path.join(stats_dir, 'error_max_xyz_{}.npy'.format(epoch)), error_max_xyz)

            error_guass_delta_now = np.mean(error_guass_delta['a1_x']) + np.mean(error_guass_delta['a1_y']) + np.mean(error_guass_delta['a1_z'])

            error_mean_max_now = np.mean(error_max_xyz['max_x']) + np.mean(error_max_xyz['max_y']) + np.mean(error_max_xyz['max_z'])

            if error_guass_delta_best < error_guass_delta_now:
                error_guass_delta_best = error_guass_delta_now

            if error_mean_max_now < error_mean_max_best:
                error_mean_max_best = error_mean_max_now

            logger.info('Best error mean:{}.  Error mean now:{}'.format(error_guass_delta_best, error_guass_delta_now))
            logger.info('Best max mean error:{}.  Error max mean now:{}'.format(error_mean_max_best, error_mean_max_now))
            if True:
                torch.save(net.state_dict(), os.path.join(stats_dir, "{}.pth".format(epoch)))
                logger.info('save pth in epoch: {}'.format(epoch))


def split_dataset(dataset, split, batch_size, num_workers, train_dataset_per, val_dataset_per):
    indices = list(range(len(dataset)))
    split = int(np.floor(split * len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]

    train_indices_choose = random.sample(train_indices, int(train_dataset_per * len(train_indices)))
    val_indices_choose = random.sample(val_indices, int(val_dataset_per * len(val_indices)))

    # Creating data samplers and loaders
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_indices_choose)
    val_sampler = torch.utils.data.sampler.SubsetRandomSampler(val_indices_choose)

    train_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        sampler=train_sampler
    )
    val_data = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        sampler=val_sampler
    )
    return train_data, val_data


if __name__ == "__main__":

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ['PYTHONHASHSEED'] = str(42)
    cudnn.benchmark = False
    cudnn.deterministic = True

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    data_path = "dataset/force_map_finite_element_7_9/"
    dataset_type = 'new'
    checkout_path = "backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
    batch_size = 48
    img_shape = (256, 256)
    epoch = 100
    lr = 0.0015
    lr_min = 0.00001
    num_workers = 0
    train_dataset_per = 1
    val_dataset_per = 1
    net_name = 'swinunet'  # transforce

    dt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
    save_folder = os.path.join('./output', dt + '_{}'.format(net_name + '_Guass'))

    stats_dir = save_folder
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    logger = log_creater(stats_dir)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = ForceData(data_path, dataset_type)

    train_dataloader, val_dataloader = split_dataset(dataset, 0.8, batch_size, num_workers, train_dataset_per, val_dataset_per)

    net = get_network(network_name=net_name, logger=logger)

    logger.info('{}'.format(net))
    train(net, criterion, train_dataloader, val_dataloader, device, batch_size, epoch, lr, lr_min, stats_dir)
