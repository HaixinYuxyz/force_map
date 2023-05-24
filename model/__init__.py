def get_network(network_name, logger):
    network_name = network_name.lower()
    if network_name == 'transforce':
        from .swin_transformer_v2.trans_forcer import TransForcer
        img_shape = (256, 256)
        checkout_path = "backbones/cifar10_swin_t_deformable_best_model_backbone.pt"
        decoder_params = {"depth": 4, "hidden_dim": 96, "norm_type": dict(type="BN"), "act_type": dict(type="LeakyReLU")}
        net = TransForcer(in_channels=6, window_size=8, input_shape=img_shape, checkout_path=checkout_path,
                          use_checkout=True, logger=logger, **decoder_params)
        return net
    elif network_name == 'unet':
        from .UNet.UNet import UNet
        net = UNet(in_channels=6, num_classes=3)
        return net
    elif network_name == 'grconvnet':
        from .grconvnet.grconvnet3 import GenerativeResnet
        net = GenerativeResnet(input_channels=6)
        return net
    elif network_name == 'resnet':
        from .ResNet_FCN.resnet_fcn import resnet_fcn
        net = resnet_fcn()
        return net
    elif network_name == 'swinunet':
        from .SwinUnet.vision_transformer import SwinUnet
        from .SwinUnet.train import get_all_config
        args, config = get_all_config()
        net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)
        return net
    elif network_name == 'fuseswinunet':
        from .Fuse_Swinunet.vision_transformer import SwinUnet
        from .Fuse_Swinunet.train import get_all_config
        args, config = get_all_config()
        net = SwinUnet(config, img_size=args.img_size, num_classes=args.num_classes)
        return net
    elif network_name == 'swintransformer_all':
        from .Swintransformer_all.vision_transformer import SwinUnet_all
        from .Swintransformer_all.train import get_all_config
        args, config = get_all_config()
        net = SwinUnet_all(config, img_size=args.img_size, num_classes=args.num_classes)
        return net
    elif network_name == 'fuse_swinunet_no_cross':
        from .Fuse_Swinunet_no_cross.vision_transformer import SwinUnet_nocross
        from .Fuse_Swinunet_no_cross.train import get_all_config
        args, config = get_all_config()
        net = SwinUnet_nocross(config, img_size=args.img_size, num_classes=args.num_classes)
        return net
