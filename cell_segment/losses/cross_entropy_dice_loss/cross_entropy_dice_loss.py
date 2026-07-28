import torch.nn as nn

from losses.multiclass_dice_loss.muticlass_dice_loss import MulticlassDiceLoss


class CrossEntropyDiceLoss(nn.Module):
    """
    Cross Entropy + Dice Loss
    """

    def __init__(
        self,
        dice_weight=1.0,
        ce_weight=1.0
    ):

        super().__init__()

        self.ce = nn.CrossEntropyLoss()

        self.dice = MulticlassDiceLoss()

        self.dice_weight = dice_weight

        self.ce_weight = ce_weight

    def forward(
        self,
        logits,
        target
    ):

        ce_loss = self.ce(
            logits,
            target
        )

        dice_loss = self.dice(
            logits,
            target
        )

        total_loss = (
            self.ce_weight * ce_loss +
            self.dice_weight * dice_loss
        )

        return total_loss