import torch


class SegmentationMetrics:
    """
    Computes multiclass segmentation metrics.

    Parameters
    ----------
    num_classes : int
        Number of segmentation classes including background.
    """

    def __init__(self, num_classes):

        self.num_classes = num_classes
        self.eps = 1e-7

    def _confusion_terms(self, pred, target, class_id):

        pred = (pred == class_id)
        target = (target == class_id)

        tp = torch.logical_and(pred, target).sum().float()

        fp = torch.logical_and(pred, ~target).sum().float()

        fn = torch.logical_and(~pred, target).sum().float()

        tn = torch.logical_and(~pred, ~target).sum().float()

        return tp, fp, fn, tn

    def dice(self, pred, target):

        dice_scores = []

        for c in range(self.num_classes):

            tp, fp, fn, _ = self._confusion_terms(
                pred,
                target,
                c
            )

            dice = (
                2 * tp + self.eps
            ) / (
                2 * tp + fp + fn + self.eps
            )

            dice_scores.append(dice)

        return torch.stack(dice_scores)

    def iou(self, pred, target):

        iou_scores = []

        for c in range(self.num_classes):

            tp, fp, fn, _ = self._confusion_terms(
                pred,
                target,
                c
            )

            iou = (
                tp + self.eps
            ) / (
                tp + fp + fn + self.eps
            )

            iou_scores.append(iou)

        return torch.stack(iou_scores)

    def precision(self, pred, target):

        scores = []

        for c in range(self.num_classes):

            tp, fp, _, _ = self._confusion_terms(
                pred,
                target,
                c
            )

            precision = (
                tp + self.eps
            ) / (
                tp + fp + self.eps
            )

            scores.append(precision)

        return torch.stack(scores)

    def recall(self, pred, target):

        scores = []

        for c in range(self.num_classes):

            tp, _, fn, _ = self._confusion_terms(
                pred,
                target,
                c
            )

            recall = (
                tp + self.eps
            ) / (
                tp + fn + self.eps
            )

            scores.append(recall)

        return torch.stack(scores)

    def evaluate(self, logits, target):
        """
        Parameters
        ----------
        logits : Tensor
            Shape (N,C,H,W)

        target : Tensor
            Shape (N,H,W)
        """

        pred = torch.argmax(
            logits,
            dim=1
        )

        dice = self.dice(
            pred,
            target
        )

        iou = self.iou(
            pred,
            target
        )

        precision = self.precision(
            pred,
            target
        )

        recall = self.recall(
            pred,
            target
        )

        results = {

            "dice_per_class":
                dice,

            "iou_per_class":
                iou,

            "precision_per_class":
                precision,

            "recall_per_class":
                recall,

            "mean_dice":
                dice.mean(),

            "mean_iou":
                iou.mean(),

            "mean_precision":
                precision.mean(),

            "mean_recall":
                recall.mean(),
        }

        return results