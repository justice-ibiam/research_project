import torch


def dice_score(pred, target):

    pred = (torch.sigmoid(pred) > 0.5).float()

    intersection = (pred * target).sum()

    dice = (
        2 * intersection + 1e-6
    ) / (
        pred.sum()
        + target.sum()
        + 1e-6
    )

    return dice.item()