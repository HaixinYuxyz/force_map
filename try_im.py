import os

import numpy as np
import cv2
from matplotlib import pyplot as plt
import torch


def plot_point_fig(idx_list, plot_error_x, plot_error_y, plot_error_z, epoch, path):
    plt.scatter(idx_list, plot_error_x)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_x)), ls=":", c="red")  # 添加水平直线
    # plt.text(len(idx_list), np.mean(np.array(plot_error_x)), 'Mean X error: {}'.format(np.mean(np.array(plot_error_x))))
    num_005x = np.sum(np.array(plot_error_x) <= 0.05)
    num_01x = np.sum(np.array(plot_error_x) <= 0.1)
    plt.title(
        'Error pred X. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005x / len(idx_list), num_01x / len(idx_list), np.mean(np.array(plot_error_x))))
    plt.savefig(os.path.join(path, 'error_x.png'))
    plt.close()

    plt.scatter(idx_list, plot_error_y)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_y)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_y)), 'Mean Y error: {}'.format(np.mean(np.array(plot_error_y))))
    num_005y = np.sum(np.array(plot_error_y) <= 0.05)
    num_01y = np.sum(np.array(plot_error_y) <= 0.1)
    plt.title(
        'Error pred Y. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005y / len(idx_list), num_01y / len(idx_list), np.mean(np.array(plot_error_y))))
    plt.savefig(os.path.join(path, 'error_y.png'))
    plt.close()

    plt.scatter(idx_list, plot_error_z)
    plt.grid(True)
    plt.axhline(y=np.mean(np.array(plot_error_z)), ls=":", c="red")
    # plt.text(len(idx_list), np.mean(np.array(plot_error_z)), 'Mean Z error: {}'.format(np.mean(np.array(plot_error_z))))
    num_005z = np.sum(np.array(plot_error_z) <= 0.05)
    num_01z = np.sum(np.array(plot_error_z) <= 0.1)
    plt.title(
        'Error pred Z. In 0.05: {:.4f}. In 0.1: {:.4f}. Mean error: {:.4f}'.format(num_005z / len(idx_list), num_01z / len(idx_list), np.mean(np.array(plot_error_z))))
    plt.savefig(os.path.join(path, 'error_z.png'))
    plt.close()

    error_x_pic = cv2.imread(os.path.join(path, 'error_x.png'))
    error_y_pic = cv2.imread(os.path.join(path, 'error_y.png'))
    error_z_pic = cv2.imread(os.path.join(path, 'error_z.png'))

    pic_save = np.hstack((error_x_pic, error_y_pic, error_z_pic))
    return pic_save


if __name__ == '__main__':
    path = '/home/shoujie/Program/force_map_new/output/23_05_13_23_12_swinunet/'
    for i in range(100):
        data_dict = np.load(os.path.join(path, 'error_{}.npy'.format(i)), allow_pickle=True).item()
        try:
            os.mkdir(os.path.join(path, 'statistical_data'))
        except:
            pass
        plot_error_x = data_dict['error x']
        plot_error_y = data_dict['error y']
        plot_error_z = data_dict['error z']
        idx_list = list(range(600))
        pic = plot_point_fig(idx_list=idx_list, plot_error_x=plot_error_x, plot_error_y=plot_error_y, plot_error_z=plot_error_z, epoch=i, path=path)
        cv2.imwrite(os.path.join(path, 'statistical_data', 'epoch_{}.png'.format(i)), pic)
