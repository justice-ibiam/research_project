import torch


def iou_score(pred, target):

    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()

    union = (
        pred
        + target
        - pred * target
    ).sum()

    return (
        (intersection + 1e-6)
        /
        (union + 1e-6)
    ).item()