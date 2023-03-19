from matplotlib import pyplot as plt
import torch
from data.data import ForceData
import matplotlib as mpl
from model.trans_forcer import TransForcer

# mpl.use('TkAgg')


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
    data_path = "src/all_new"
    checkout_path = "output/23_03_04_11_08/99.pth"
    checkpoint = torch.load(checkout_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    img_shape = (256, 256)
    decoder_params = {"depth": 4, "hidden_dim": 96, "norm_type": dict(type="BN"), "act_type": dict(type="LeakyReLU")}
    net = TransForcer(in_channels=3, window_size=8, input_shape=img_shape, checkout_path=checkout_path,
                      use_checkout=False, **decoder_params)
    net.load_state_dict(checkpoint)
    net.eval()
    net.to(device)

    dataset = ForceData(data_path)
    rgb, inf, targets, mask = dataset[25]

    # rgb = rgb.unsqueeze(0)
    # rgb = rgb.to(device)
    inf = inf.unsqueeze(0)
    inf = inf.to(device)
    mask = mask.to(device)
    for target in targets:
        target.to(device)

    force_maps = net(inf)
    errorx = abs(torch.max(targets[0]) - torch.max(force_maps[0]))
    errory = abs(torch.max(targets[1]) - torch.max(force_maps[1]))
    errorz = abs(torch.max(targets[2]) - torch.max(force_maps[2]))

    print('X label:----->', abs(torch.max(targets[0]) - torch.max(force_maps[0])))
    print('Y label:----->', abs(torch.max(targets[1]) - torch.max(force_maps[1])))
    print('Z label:----->', abs(torch.max(targets[2]) - torch.max(force_maps[2])))

    vis_forcemap(inf, inf, targets, mask, force_maps)
