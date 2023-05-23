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

    optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr, betas=(0.9, 0.999), weight_decay=0.01)
    warm_up_with_multistep_lr = lambda epoch: epoch / (20 * 225) if epoch <= (20 * 225) else 0.4 ** len(
        [m for m in [40 * 225, 60 * 225, 80 * 225, 90 * 225] if m <= epoch])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_multistep_lr)

    tb = tensorboardX.SummaryWriter(stats_dir)

    total_iter_num = 0
    val_total_iter_num = 0
    error_mean_best = 1000
    for epoch in range(num_epoch):
        print("——————{} epoch——————".format(epoch + 1))

        net.train()
        for batch in tqdm(train_dataloader, desc='training'):
            rgb, inf, targets, mask, _ = batch
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
        idx_list = []
        with torch.no_grad():
            for idx, batch in tqdm(enumerate(valid_dataloader), desc='valling'):
                rgb, inf, targets, mask, real_force = batch
                real_force = [force.to(device) for force in real_force]
                rgb = rgb.to(device)
                inf = inf.to(device)
                mask = mask.to(device)
                labels = [target.to(device) for target in targets]
                output = net(rgb, inf)
                loss_list, Loss = criterion(output, labels)

                tb.add_scalar('val loss/sum_loss', Loss.item(), val_total_iter_num)
                tb.add_scalar('val loss/x_loss', loss_list[0].item(), val_total_iter_num)
                tb.add_scalar('val loss/y_loss', loss_list[1].item(), val_total_iter_num)
                tb.add_scalar('val loss/z_loss', loss_list[2].item(), val_total_iter_num)

                real_x, real_y, real_z = real_force

                pred_x = (torch.sum(mask * output[0]) / torch.sum(mask)) * (1.1671 + 1.4656) - 1.4656
                pred_y = (torch.sum(mask * output[1]) / torch.sum(mask)) * (0.9182 + 1.6964) - 0.9182
                pred_z = -((torch.sum(mask * output[2]) / torch.sum(mask)) * (4.0909 - 1.5137) + 1.5137)

                error_x = torch.abs(real_x - pred_x)
                error_y = torch.abs(real_y - pred_y)
                error_z = torch.abs(real_z - pred_z)

                plot_error_x.append(error_x.item())
                plot_error_y.append(error_y.item())
                plot_error_z.append(error_z.item())
                idx_list.append(idx)

                if val_total_iter_num % 20 == 0:
                    grid_image = create_grid_image(rgb=rgb, inf=inf, mask=mask, output=output, labels=labels, max_num_images_to_save=6)
                    tb.add_image('Val image', grid_image, val_total_iter_num)
                val_total_iter_num += 1

            save_dict = {'error x': np.array(plot_error_x), 'error y': np.array(plot_error_y), 'error z': np.array(plot_error_z)}

            np.save(os.path.join(stats_dir, 'error_{}.npy'.format(epoch)), save_dict)

            plot_error_x_np = np.array(plot_error_x)
            plot_error_y_np = np.array(plot_error_y)
            plot_error_z_np = np.array(plot_error_z)

            error_mean_now = np.mean(plot_error_x_np) + np.mean(plot_error_y_np) + np.mean(plot_error_z_np)

            tb.add_scalar('val error/error mean all', error_mean_now, epoch)
            tb.add_scalar('val error/error mean x', np.mean(plot_error_x_np), epoch)
            tb.add_scalar('val error/error mean y', np.mean(plot_error_y_np), epoch)
            tb.add_scalar('val error/error mean z', np.mean(plot_error_z_np), epoch)

            if error_mean_now < error_mean_best:
                error_mean_best = error_mean_now
                torch.save(net.state_dict(), os.path.join(stats_dir, "best.pth"))
                logger.info('save best pth in epoch: {}'.format(epoch))
            logger.info('Best error mean:{}.  Error mean now:{}'.format(error_mean_best, error_mean_now))
            # if epoch % 10 == 0:
            if True:
                torch.save(net.state_dict(), os.path.join(stats_dir, "{}.pth".format(epoch)))
                logger.info('save pth in epoch: {}'.format(epoch))
            error_x_pic, error_y_pic, error_z_pic = plot_point_fig(idx_list, plot_error_x, plot_error_y, plot_error_z, tb, epoch, logger)
            tb.add_image('Error_statistics/error_x', error_x_pic, epoch)
            tb.add_image('Error_statistics/error_y', error_y_pic, epoch)
            tb.add_image('Error_statistics/error_z', error_z_pic, epoch)


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
    data_path = "dataset/force_mask"
    checkout_path = "backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
    batch_size = 24
    img_shape = (256, 256)
    epoch = 100
    lr = 0.001
    lr_min = 0.00001
    num_workers = 0
    train_dataset_per = 1
    val_dataset_per = 1
    net_name = 'fuseswinunet'  # transforce

    dt = datetime.datetime.now().strftime('%y_%m_%d_%H_%M')
    save_folder = os.path.join('./output', dt + '_{}'.format(net_name))

    stats_dir = save_folder
    if not os.path.exists(stats_dir):
        os.makedirs(stats_dir)
    logger = log_creater(stats_dir)

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    dataset = ForceData(data_path)

    train_dataloader, val_dataloader = split_dataset(dataset, 0.9, batch_size, num_workers, train_dataset_per, val_dataset_per)

    net = get_network(network_name=net_name, logger=logger)

    logger.info('{}'.format(net))
    train(net, criterion, train_dataloader, val_dataloader, device, batch_size, epoch, lr, lr_min, stats_dir)
