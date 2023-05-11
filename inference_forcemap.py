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
    data_path = "dataset/force_mask"
    checkout_path = "/home/shoujie/Program/force_map_new/output/23_05_11_13_48_transforce/best.pth"
    checkpoint = torch.load(checkout_path)
    logger = log_creater('/home/shoujie/Program/force_map_new/output/23_05_11_13_48_transforce')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_shape = (256, 256)
    decoder_params = {"depth": 4, "hidden_dim": 96, "norm_type": dict(type="BN"), "act_type": dict(type="LeakyReLU")}
    net = TransForcer(in_channels=6, window_size=8, input_shape=img_shape, checkout_path=checkout_path,
                      use_checkout=False, logger=logger, **decoder_params)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)

    dataset = ForceData(data_path)

    indices = list(range(len(dataset)))
    split = int(np.floor(0.9 * len(dataset)))
    np.random.shuffle(indices)
    train_indices, val_indices = indices[:split], indices[split:]
    plot_error_x = []
    plot_error_y = []
    plot_error_z = []
    idx_list = []
    # val_indices = val_indices[0:20]
    for id, num in tqdm(enumerate(val_indices)):
        rgb, inf, labels, mask, real_force = dataset[num]
        mask = mask.to(device)
        rgb = rgb.unsqueeze(0)
        rgb = rgb.to(device)
        inf = inf.unsqueeze(0)
        inf = inf.to(device)
        mask = mask.to(device)
        for label in labels:
            label.to(device)

        output = net(rgb, inf)

        error_cal_x = torch.abs((torch.sum(mask * labels[0].to(device)) / torch.sum(mask)) - (torch.sum(mask * output[0]) / torch.sum(mask))) * (1.1671 + 1.4656)
        error_cal_y = torch.abs((torch.sum(mask * labels[1].to(device)) / torch.sum(mask)) - (torch.sum(mask * output[1]) / torch.sum(mask))) * (0.9182 + 1.6964)
        error_cal_z = torch.abs((torch.sum(mask * labels[2].to(device)) / torch.sum(mask)) - (torch.sum(mask * output[2]) / torch.sum(mask))) * (4.0909 - 1.5137)

        real_x, real_y, real_z = real_force

        pred_x = (torch.sum(mask * output[0]) / torch.sum(mask)) * (1.1671 + 1.4656) - 1.4656
        pred_y = (torch.sum(mask * output[1]) / torch.sum(mask)) * (0.9182 + 1.6964) - 0.9182
        pred_z = -((torch.sum(mask * output[2]) / torch.sum(mask)) * (4.0909 - 1.5137) + 1.5137)

        logger.info('Real X: {} Pred X: {}'.format(real_x, pred_x))
        logger.info('Real Y: {} Pred Y: {}'.format(real_y, pred_y))
        logger.info('Real Z: {} Pred Z: {}'.format(real_z, pred_z))

        error_x = torch.abs(real_x - pred_x)
        error_y = torch.abs(real_y - pred_y)
        error_z = torch.abs(real_z - pred_z)

        plot_error_x.append(error_x.item())
        plot_error_y.append(error_y.item())
        plot_error_z.append(error_z.item())
        idx_list.append(id)

    # vis_forcemap(inf, inf, labels, mask, output)
    plt.scatter(idx_list, plot_error_x)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_x)), ls=":", c="red")  # 添加水平直线
    # plt.text(len(idx_list), np.mean(np.array(plot_error_x)), 'Mean X error: {}'.format(np.mean(np.array(plot_error_x))))
    num_005x = np.sum(np.array(plot_error_x) <= 0.05)
    num_01x = np.sum(np.array(plot_error_x) <= 0.1)
    plt.title('Error pred X. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005x / len(idx_list), num_01x / len(idx_list), np.mean(np.array(plot_error_x))))
    plt.savefig('error_x.png')
    plt.close()

    plt.scatter(idx_list, plot_error_y)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_y)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_y)), 'Mean Y error: {}'.format(np.mean(np.array(plot_error_y))))
    num_005y = np.sum(np.array(plot_error_y) <= 0.05)
    num_01y = np.sum(np.array(plot_error_y) <= 0.1)
    plt.title('Error pred Y. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005y / len(idx_list), num_01y / len(idx_list), np.mean(np.array(plot_error_y))))
    plt.savefig('error_y.png')
    plt.close()

    plt.scatter(idx_list, plot_error_z)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_z)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_z)), 'Mean Z error: {}'.format(np.mean(np.array(plot_error_z))))
    num_005z = np.sum(np.array(plot_error_z) <= 0.05)
    num_01z = np.sum(np.array(plot_error_z) <= 0.1)
    plt.title('Error pred Z. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005z / len(idx_list), num_01z / len(idx_list), np.mean(np.array(plot_error_z))))
    plt.savefig('error_z.png')
