# Directly copied from:
# https://github.com/DmitryUlyanov/deep-image-prior
# NOTE: This is a modified version of the original code. We removed unused
# functions and classes.

import torch.nn as nn

from .unet import UNet

def get_net(input_depth, NET_TYPE, pad, upsample_mode):
    assert NET_TYPE == "UNet"
    net = UNet(
        num_input_channels=input_depth,
        num_output_channels=3,
        feature_scale=4,
        more_layers=0,
        concat_x=False,
        upsample_mode=upsample_mode,
        pad=pad,
        norm_layer=nn.BatchNorm2d,
        need_sigmoid=True,
        need_bias=True
    )
    return net