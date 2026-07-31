import torch.nn as nn
import torch
from omegaconf import DictConfig
from omegaconf.base import ContainerMetadata, Metadata
from omegaconf.nodes import AnyNode
import typing
import collections
from .unet.unet import UNet
from .u2net.u2net import U2NET
from torchvision.models.segmentation import deeplabv3_resnet50
from .modified_unet import UNetM





def get_model(cfg):
    if cfg.model.name == "unet":
        model = UNet()
        return model
    elif cfg.model.name == "u2net":
        return U2NET(
            in_ch=3,
            # out_ch=cfg.dataset.num_classes,
            out_ch=13,
        )
    elif cfg.model.name == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(
            weights=None,
            num_classes=1
        )
        return model
    elif cfg.model.name == "unetm":
        model = UNetM()
        return model       


