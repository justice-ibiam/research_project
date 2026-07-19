import torch
import torch.nn as nn

class DiceLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        pred,
        target
    ):

        pred = torch.sigmoid(pred)

        pred = pred.view(-1)
        target = target.view(-1)

        intersection = (
            pred*target
        ).sum()

        dice = (
            2*intersection + 1
        ) / (
            pred.sum()
            + target.sum()
            + 1
        )

        return 1-dice