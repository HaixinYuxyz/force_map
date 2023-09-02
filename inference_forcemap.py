from matplotlib import pyplot as plt
import torch
from data.data import ForceData
from model.swin_transformer_v2.trans_forcer import TransForcer
import logging
import os
import numpy as np
import torch.backends.cudnn as cudnn
import random
from tqdm import tqdm
from model import get_network
from main import split_dataset
from utils.percentage_analyze import percentage_analyze
from main import sum_mean_acc


# mpl.use('TkAgg')

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


def vis_forcemap(rgb, inf, targets, mask, force_maps):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(3, 3, 1)
    ax.imshow(rgb.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    ax.set_title("rgb")

    ax = fig.add_subplot(3, 3, 2)
    ax.imshow(inf.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
    ax.set_title("inf")

    ax = fig.add_subplot(3, 3, 3)
    ax.imshow(mask.cpu().detach().numpy().transpose(1, 2, 0) * 255)
    ax.set_title("mask")

    ax = fig.add_subplot(3, 3, 4)
    ax.imshow(targets[0].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("x_origin")

    ax = fig.add_subplot(3, 3, 5)
    ax.imshow(targets[1].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("y_origin")

    ax = fig.add_subplot(3, 3, 6)
    ax.imshow(targets[2].cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("z_origin")

    ax = fig.add_subplot(3, 3, 7)
    ax.imshow(force_maps[0].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0,
              vmax=255)
    ax.set_title("x_predict")

    ax = fig.add_subplot(3, 3, 8)
    ax.imshow(force_maps[1].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0)[:, :, 0] * 255, cmap='jet', vmin=0,
              vmax=255)
    ax.set_title("y_predict")

    ax = fig.add_subplot(3, 3, 9)
    ax.imshow(force_maps[2].squeeze(0).cpu().detach().numpy().transpose(1, 2, 0) * 255, cmap='jet', vmin=0, vmax=255)
    ax.set_title("z_predict")

    fig.savefig("result480.png")


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

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_path = "dataset/force_map_finite_element_7_9/"
    checkout_path = "/home/shoujie/Program/force_map_new/output/23_08_29_20_59_swinunet_Guass/99.pth"
    checkpoint = torch.load(checkout_path)
    logger = log_creater("/home/shoujie/Program/force_map_new/output/23_08_28_15_36_swinunet_Guass")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net_name = 'swinunet'  # transforce
    net = get_network(network_name=net_name, logger=logger)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)
    batch_size = 48
    img_shape = (256, 256)
    epoch = 100
    lr = 0.001
    lr_min = 0.00001
    num_workers = 0
    train_dataset_per = 1
    val_dataset_per = 1
    net_name = 'swinunet'  # transforce

    dataset = ForceData(data_path, 'new')

    train_dataloader, val_dataloader = split_dataset(dataset, 0.8, batch_size, num_workers, train_dataset_per, val_dataset_per)
    all_error = 0
    # for percent in [0.00005, 0.0001, 0.0002, 0.0003, 0.0004, 0.0005, 0.0006, 0.0007, 0.0008, 0.0009, 0.001, 0.002]:
    x_list = []
    y_list = []
    z_list = []
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(val_dataloader), desc='valling'):
            rgb, inf, targets, mask, real_force, d_type, max_min_force, max_xyz, force_maxs = batch
            real_force = [force.to(device) for force in real_force]
            max_xyz = max_xyz.to(device)
            max_min_force = [max_min.to(device) for max_min in max_min_force]
            inf = inf.to(device)
            mask = mask.to(device)
            labels = [target.to(device) for target in targets]
            output_map = net(inf)
            orin_map, max_map = output_map
            error_x, error_y, error_z = sum_mean_acc(max_map, force_maxs, mask, max_min_force)
            x_list.append(error_x.item())
            y_list.append(error_y.item())
            z_list.append(error_z.item())
    print(np.mean(np.array(x_list)), np.mean(np.array(y_list)), np.mean(np.array(x_list)))
