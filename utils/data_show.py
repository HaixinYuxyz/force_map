import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use('TkAgg')


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


orin_data = get_txt_data(r'src/all_new/data.txt')

force_x = orin_data[:, 0]
force_y = orin_data[:, 1]
force_z = orin_data[:, 2]

x = np.arange(0, len(force_x), 1)

# plt.plot(x, force_x)
# plt.plot(x, force_y)
plt.plot(x, force_z)
plt.show()
