import os

import numpy as np
from matplotlib import pyplot as plt


def plot_img(delta_dict, x_list, save_path, label, save_name, epoch):
    plt.figure(figsize=(48, 48))
    plt.subplot(3, 3, 1)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a1_x'], marker='D', color='g', label='a1_x')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a1_x'])]
    plt.title(label='a1_x : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a1_x']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 2)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a2_x'], marker='D', color='g', label='a2_x')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a2_x'])]
    plt.title(label='a2_x : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a2_x']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 3)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a3_x'], marker='D', color='g', label='a3_x')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a3_x'])]
    plt.title(label='a3_x : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a3_x']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 4)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a1_y'], marker='D', color='g', label='a1_y')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a1_y'])]
    plt.title(label='a1_y : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a1_y']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 5)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a2_y'], marker='D', color='g', label='a2_y')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a2_y'])]
    plt.title(label='a2_y : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a2_y']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 6)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a3_y'], marker='D', color='g', label='a3_y')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a3_y'])]
    plt.title(label='a3_y : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a3_y']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 7)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a1_z'], marker='D', color='g', label='a1_z')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a1_z'])]
    plt.title(label='a1_z : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a1_z']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 8)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a2_z'], marker='D', color='g', label='a2_z')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a2_z'])]
    plt.title(label='a2_z : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a2_z']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.subplot(3, 3, 9)
    plt.ylim((0, 1))
    plt.scatter(x_list, delta_dict['a3_z'], marker='D', color='g', label='a3_z')
    plt.grid(True)
    sum_dict = [i * j for (i, j) in zip(delta_dict['mask_pix'], delta_dict['a3_z'])]
    plt.title(label='a3_z : Mean acc = {}, Mean acc(mask) = {} '.format(np.mean(delta_dict['a3_z']),
                                                                        np.sum(sum_dict) / np.sum(delta_dict['mask_pix'])))

    plt.suptitle("RUNOOB subplot Test")
    # plt.show()

    # plt.figure(figsize=(24, 8))

    # plt.grid(True)
    # plt.title(label=save_name)
    plt.savefig(os.path.join(save_path, '{}.png'.format(save_name)), dpi=400)
    plt.close()


if __name__ == '__main__':
    data_path = r"/home/shoujie/Program/force_map_new/output/23_07_27_23_38_swinunet_Guass/"
    img_save_path = os.path.join(data_path, 'delta_save')
    if not os.path.exists(img_save_path):
        os.makedirs(img_save_path)
    x_list = np.arange(9600).tolist()
    for i in range(0, 100):
        print('----------------{}'.format(i))
        delta_name = os.path.join(data_path, 'error_guass_delta_{}.npy'.format(i))
        delta_dict = np.load(delta_name, allow_pickle=True).item()

        plot_img(delta_dict, x_list, img_save_path, 0, i, 0)
