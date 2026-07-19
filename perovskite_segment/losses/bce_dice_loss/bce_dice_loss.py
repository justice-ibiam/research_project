import torch.nn as nn

from losses.dice_loss.dice_loss import DiceLoss


class BCEDiceLoss(nn.Module):

    def __init__(self):

        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        pred,
        target
    ):

        return (
            self.bce(pred, target)
            +
            self.dice(pred, target)
        )