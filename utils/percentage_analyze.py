import numpy as np


def percentage_analyze(x_map, y_map, z_map, max_force, percent):
    pic_shape = x_map.size
    x_map = np.abs(x_map)
    y_map = np.abs(y_map)
    z_map = np.abs(z_map)

    x_max_pred = np.mean(np.sort(x_map.flatten())[::-1][:int(pic_shape * percent)])
    y_max_pred = np.mean(np.sort(y_map.flatten())[::-1][:int(pic_shape * percent)])
    z_max_pred = np.mean(np.sort(z_map.flatten())[::-1][:int(pic_shape * percent)])
    x_error = np.abs(x_max_pred - max_force[0][0].item())
    y_error = np.abs(y_max_pred - max_force[0][1].item())
    z_error = np.abs(z_max_pred - max_force[0][2].item())

    return x_error, y_error, z_error
