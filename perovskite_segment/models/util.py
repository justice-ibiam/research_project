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





def get_model(cfg):
    if cfg.model.name == "unet":
        model = UNet()
        return model
    elif cfg.model.name == "u2net":
        return U2NET(
            in_ch=3,
            out_ch=1,
        )
    elif cfg.model.name == 'deeplabv3_resnet50':
        model = deeplabv3_resnet50(
            weights=None,
            num_classes=1
        )
        return model


def set_trainable_layers(model: nn.Module, mode: str = "fc", model_name="resnet50"):
    """
    Set which parts of ResNet50, EfficientNet-B0, and DenseNet121 are trainable.
    
    mode: "fc", "classifier", "last_block", "all"
    model_name: "resnet50", "efficientnet_b0", "densenet121"
    """

    assert mode in ["fc", "classifier", "last_block", "all"], f"Unknown mode: {mode}"
    print(f"mode: {mode} model_name: {model_name} model: {model}")
    # Freeze everything first
    for param in model.parameters():
        param.requires_grad = False


    # FC / Classifier only
    if mode in ["fc", "classifier"]:
        if model_name == "resnet50":
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True
        elif model_name == "densenet121":
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True
        elif model_name == "efficientnet_b0":
            # EfficientNet classifier is Sequential: [Dropout, Linear]
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True
        elif model_name in ["own_cnn", "OwnCNN"]:
            print("Freezing base convolutional layer")
            for param in model.base_layer.parameters():
                param.requires_grad = False
    
            #freezing the rest of the layers
            print("Freezing the rest of the convolutional layers")
            for param in model.conv_layers.parameters():
                param.requires_grad = False

            #unfreeze the classifier
            print("Unfreezing the classifier")
            for layer in [model.lin_1, model.lin_2, model.lin_3]:
                for param in layer.parameters():
                    param.requires_grad = True
                   

        print("Training classifier only.")
        return

    # Last block + classifier
    if mode == "last_block":
        if model_name == "resnet50":
            for p in model.backbone.layer4.parameters():
                p.requires_grad = True
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True

        elif model_name == "densenet121":
            # DenseNet last dense block is denseblock4
            for p in model.backbone.features.denseblock4.parameters():
                p.requires_grad = True
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True

        elif model_name == "efficientnet_b0":
            # Last EfficientNet block is at index -2 inside model.features
            last_block = model.backbone.features[-2]  
            for p in last_block.parameters():
                p.requires_grad = True
            for p in model.ordinal_fc.parameters():
                p.requires_grad = True
                

        print("Training last block + classifier.")
        return


    # Full Finetuning
    if mode == "all":
        for param in model.parameters():
            param.requires_grad = True
        print("Training ALL layers.")

def get_checkpoint(model, device, checkpoint_path):

    safe_types = [
        DictConfig,
        ContainerMetadata,
        Metadata,
        AnyNode,
        typing.Any,
        dict,
        tuple,
        list,
        set,
        frozenset,
        collections.defaultdict,
    ]

    with torch.serialization.safe_globals(safe_types):
        checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state"])
    return 

