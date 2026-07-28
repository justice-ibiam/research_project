import torch
import torch.nn as nn
import torch.nn.functional as F

class MulticlassDiceLoss(nn.Module):
    """
    Dice loss for multiclass semantic segmentation.

    Prediction:
        (N, C, H, W)

    Target:
        (N, H, W)

    where
        N = batch size
        C = number of classes
    """

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, target):

        num_classes = logits.shape[1]

        # probabilities
        probs = F.softmax(logits, dim=1)

        # one-hot encode target
        target_onehot = F.one_hot(
            target,
            num_classes=num_classes
        )

        target_onehot = target_onehot.permute(
            0,
            3,
            1,
            2
        ).float()

        dims = (0, 2, 3)

        intersection = torch.sum(
            probs * target_onehot,
            dim=dims
        )

        union = torch.sum(
            probs,
            dim=dims
        ) + torch.sum(
            target_onehot,
            dim=dims
        )

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            union + self.smooth
        )

        return 1 - dice.mean()