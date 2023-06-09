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


def sum_mean_acc():
    pass


def criterion(pred, gt):
    losses = []
    for y, yt in zip(pred, gt):
        losses.append(F.smooth_l1_loss(y, yt))
    loss_sum = reduce(lambda x, y: x + y, losses)

    return losses, loss_sum


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
    val_total_iter_num = 0
    error_guass_max_mean_best = 1000
    for epoch in range(num_epoch):
        print("——————{} epoch——————".format(epoch + 1))

        net.train()
        for batch in tqdm(train_dataloader, desc='training'):
            rgb, inf, targets, mask, mask_gauss, real_force, d_type, max_min_force = batch
            rgb = rgb.to(device)
            inf = inf.to(device)
            labels = [target.to(device) for target in targets]
            output = net(rgb, inf)
            loss_list, Loss = criterion(output, labels)
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            tb.add_scalar('train loss/sum_loss', Loss.item(), total_iter_num)
            tb.add_scalar('train loss/x_loss', loss_list[0].item(), total_iter_num)
            tb.add_scalar('train loss/y_loss', loss_list[1].item(), total_iter_num)
            tb.add_scalar('train loss/z_loss', loss_list[2].item(), total_iter_num)

            current_learning_rate = optimizer.param_groups[0]['lr']
            tb.add_scalar('Learning_Rate', current_learning_rate, total_iter_num)

            if total_iter_num % 20 == 0:
                grid_image = create_grid_image(rgb=rgb, inf=inf, mask=mask, output=output, labels=labels, max_num_images_to_save=6)
                tb.add_image('Train image', grid_image, total_iter_num)
            total_iter_num += 1
            scheduler.step()

        # =============================================================================================================
        net.eval()
        plot_error_x = []
        plot_error_y = []
        plot_error_z = []

        error_guass_max = {'error_x': [], 'error_y': [], 'error_z': []}

        error_guass_delta = {'mask_pix': [], 'a1_x': [], 'a2_x': [], 'a3_x': [], 'a1_y': [], 'a2_y': [], 'a3_y': [], 'a1_z': [], 'a2_z': [], 'a3_z': []}

        idx_list = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_dataloader), desc='valling'):
                rgb, inf, targets, mask, mask_gauss, real_force, d_type, max_min_force = batch
                real_force = [force.to(device) for force in real_force]
                max_min_force = [max_min.to(device) for max_min in max_min_force]
                rgb = rgb.to(device)
                inf = inf.to(device)
                mask = mask.to(device)
                labels = [target.to(device) for target in targets]
                output = net(rgb, inf)

                error_guass_max_x, error_guass_max_y, error_guass_max_z = guass_map_cal_max_force(output, real_force, mask, max_min_force)
                error_guass_max['error_x'].append(error_guass_max_x.item())
                error_guass_max['error_y'].append(error_guass_max_y.item())
                error_guass_max['error_z'].append(error_guass_max_z.item())
                a1_x, a2_x, a3_x, a1_y, a2_y, a3_y, a1_z, a2_z, a3_z = guass_map_cal_delta(output, labels, mask, max_min_force)

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

                # pred_x = (torch.sum(mask * output[0]) / torch.sum(mask)) * (max_min_force[0] - max_min_force[1]) + max_min_force[1]
                # pred_y = (torch.sum(mask * output[1]) / torch.sum(mask)) * (max_min_force[2] - max_min_force[3]) + max_min_force[3]
                # pred_z = (torch.sum(mask * output[2]) / torch.sum(mask)) * (max_min_force[4] - max_min_force[5]) + max_min_force[5]

                # error_x = torch.abs(real_x - pred_x)
                # error_y = torch.abs(real_y - pred_y)
                # error_z = torch.abs(real_z - pred_z)

                # plot_error_x.append(error_x.item())
                # plot_error_y.append(error_y.item())
                # plot_error_z.append(error_z.item())
                # idx_list.append(idx)

                # if val_total_iter_num % 20 == 0:
                #     grid_image = create_grid_image(rgb=rgb, inf=inf, mask=mask, output=output, labels=labels, max_num_images_to_save=6)
                #     tb.add_image('Val image', grid_image, val_total_iter_num)
                # val_total_iter_num += 1

            # save_dict = {'error x': np.array(plot_error_x), 'error y': np.array(plot_error_y), 'error z': np.array(plot_error_z)}

            # np.save(os.path.join(stats_dir, 'error_{}.npy'.format(epoch)), save_dict)
            np.save(os.path.join(stats_dir, 'error_guass_max_{}.npy'.format(epoch)), error_guass_max)
            np.save(os.path.join(stats_dir, 'error_guass_delta_{}.npy'.format(epoch)), error_guass_delta)

            error_guass_max_mean_now = np.mean(error_guass_max['error_x']) + np.mean(error_guass_max['error_y']) + np.mean(error_guass_max['error_z'])

            if error_guass_max_mean_best > error_guass_max_mean_now:
                error_guass_max_mean_best = error_guass_max_mean_now

            logger.info('Best error mean:{}.  Error mean now:{}'.format(error_guass_max_mean_best, error_guass_max_mean_now))
            # plot_error_x_np = np.array(plot_error_x)
            # plot_error_y_np = np.array(plot_error_y)
            # plot_error_z_np = np.array(plot_error_z)

            # error_mean_now = np.mean(plot_error_x_np) + np.mean(plot_error_y_np) + np.mean(plot_error_z_np)

            # tb.add_scalar('val error/error mean all', error_mean_now, epoch)
            # tb.add_scalar('val error/error mean x', np.mean(plot_error_x_np), epoch)
            # tb.add_scalar('val error/error mean y', np.mean(plot_error_y_np), epoch)
            # tb.add_scalar('val error/error mean z', np.mean(plot_error_z_np), epoch)

            # if error_mean_now < error_mean_best:
            #     error_mean_best = error_mean_now
            #     torch.save(net.state_dict(), os.path.join(stats_dir, "best.pth"))
            #     logger.info('save best pth in epoch: {}'.format(epoch))
            # logger.info('Best error mean:{}.  Error mean now:{}'.format(error_mean_best, error_mean_now))
            # if epoch % 10 == 0:
            if True:
                torch.save(net.state_dict(), os.path.join(stats_dir, "{}.pth".format(epoch)))
                logger.info('save pth in epoch: {}'.format(epoch))
            # error_x_pic, error_y_pic, error_z_pic = plot_point_fig(idx_list, plot_error_x, plot_error_y, plot_error_z, tb, epoch, logger)
            # tb.add_image('Error_statistics/error_x', error_x_pic, epoch)
            # tb.add_image('Error_statistics/error_y', error_y_pic, epoch)
            # tb.add_image('Error_statistics/error_z', error_z_pic, epoch)


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
    data_path = "dataset/Data_65"
    dataset_type = 'old'
    checkout_path = "backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
    batch_size = 48
    img_shape = (256, 256)
    epoch = 100
    lr = 0.001
    lr_min = 0.00001
    num_workers = 0
    train_dataset_per = 1
    val_dataset_per = 1
    net_name = 'fuseswinunet'  # transforce

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
