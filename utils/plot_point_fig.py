from matplotlib import pyplot as plt
import numpy as np
import cv2
import torch


def plot_point_fig(idx_list, plot_error_x, plot_error_y, plot_error_z, tb, epoch, logger):
    plt.scatter(idx_list, plot_error_x)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_x)), ls=":", c="red")  # 添加水平直线
    # plt.text(len(idx_list), np.mean(np.array(plot_error_x)), 'Mean X error: {}'.format(np.mean(np.array(plot_error_x))))
    num_005x = np.sum(np.array(plot_error_x) <= 0.05)
    num_01x = np.sum(np.array(plot_error_x) <= 0.1)
    tb.add_scalar('Acc x/0.1', num_01x / len(idx_list), epoch)
    tb.add_scalar('Acc x/0.05', num_005x / len(idx_list), epoch)
    logger.info('Acc X 0.05: {:.4f} Acc X 0.1: {:.4f}'.format(num_005x / len(idx_list), num_01x / len(idx_list)))
    plt.title(
        'Error pred X. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005x / len(idx_list), num_01x / len(idx_list), np.mean(np.array(plot_error_x))))
    plt.savefig('error_x.png')
    plt.close()

    plt.scatter(idx_list, plot_error_y)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_y)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_y)), 'Mean Y error: {}'.format(np.mean(np.array(plot_error_y))))
    num_005y = np.sum(np.array(plot_error_y) <= 0.05)
    num_01y = np.sum(np.array(plot_error_y) <= 0.1)
    tb.add_scalar('Acc y/0.1', num_01y / len(idx_list), epoch)
    tb.add_scalar('Acc y/0.05', num_005y / len(idx_list), epoch)
    logger.info('Acc Y 0.05: {:.4f} Acc Y 0.1: {:.4f}'.format(num_005y / len(idx_list), num_01y / len(idx_list)))
    plt.title(
        'Error pred Y. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005y / len(idx_list), num_01y / len(idx_list), np.mean(np.array(plot_error_y))))
    plt.savefig('error_y.png')
    plt.close()

    plt.scatter(idx_list, plot_error_z)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_z)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_z)), 'Mean Z error: {}'.format(np.mean(np.array(plot_error_z))))
    num_005z = np.sum(np.array(plot_error_z) <= 0.05)
    num_01z = np.sum(np.array(plot_error_z) <= 0.1)
    tb.add_scalar('Acc z/0.1', num_01z / len(idx_list), epoch)
    tb.add_scalar('Acc z/0.05', num_005z / len(idx_list), epoch)
    logger.info('Acc Z 0.05: {:.4f} Acc Z 0.1: {:.4f}'.format(num_005z / len(idx_list), num_01z / len(idx_list)))
    plt.title(
        'Error pred Z. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005z / len(idx_list), num_01z / len(idx_list), np.mean(np.array(plot_error_z))))
    plt.savefig('error_z.png')
    plt.close()

    error_x_pic = torch.tensor(cv2.imread('error_x.png')).permute(2, 0, 1)
    error_y_pic = torch.tensor(cv2.imread('error_y.png')).permute(2, 0, 1)
    error_z_pic = torch.tensor(cv2.imread('error_z.png')).permute(2, 0, 1)

    return error_x_pic, error_y_pic, error_z_pic
