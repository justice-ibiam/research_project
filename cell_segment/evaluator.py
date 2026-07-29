import torch

from losses.cross_entropy_dice_loss.cross_entropy_dice_loss import CrossEntropyDiceLoss
from metrics.segmentation_metrics import SegmentationMetrics


class Evaluator:

    def __init__(
        self,
        cfg,
        eval_loader,
        model,
        device
    ):

        self.cfg = cfg

        self.eval_loader = eval_loader

        self.model = model

        self.device = device

        self.criterion = CrossEntropyDiceLoss()

        self.metrics = SegmentationMetrics(
            num_classes=cfg.dataset.num_classes
        )

    @torch.no_grad()
    def eval(self):

        self.model.eval()

        total_loss = 0.0

        total_mean_dice = 0.0
        total_mean_iou = 0.0
        total_mean_precision = 0.0
        total_mean_recall = 0.0

        total_fg_dice = 0.0
        total_fg_iou = 0.0

        n_batches = 0

        for images, masks in self.eval_loader:

            images = images.to(self.device)
            masks = masks.to(self.device)

            outputs = self.model(images)

            

            # Handle different model outputs


            if self.cfg.model.name == "deeplabv3_resnet50":

                logits = outputs["out"]

                loss = self.criterion(
                    logits,
                    masks
                )

            else:

                if isinstance(outputs, (tuple, list)):

                    loss = torch.stack([
                        self.criterion(
                            pred,
                            masks
                        )
                        for pred in outputs
                    ]).mean()

                    logits = outputs[0]

                else:

                    logits = outputs

                    loss = self.criterion(
                        logits,
                        masks
                    )

            total_loss += loss.item()

            batch_metrics = self.metrics.evaluate(
                logits,
                masks
            )

            total_mean_dice += (
                batch_metrics["mean_dice"].item()
            )

            total_mean_iou += (
                batch_metrics["mean_iou"].item()
            )

            total_mean_precision += (
                batch_metrics["mean_precision"].item()
            )

            total_mean_recall += (
                batch_metrics["mean_recall"].item()
            )

            total_fg_dice += (
                batch_metrics["dice_per_class"][1:]
                .mean()
                .item()
            )

            total_fg_iou += (
                batch_metrics["iou_per_class"][1:]
                .mean()
                .item()
            )

            n_batches += 1

        metrics = {

            "val_loss":
                total_loss / n_batches,

            "val_mean_dice":
                total_mean_dice / n_batches,

            "val_mean_iou":
                total_mean_iou / n_batches,

            "val_mean_precision":
                total_mean_precision / n_batches,

            "val_mean_recall":
                total_mean_recall / n_batches,

            "val_fg_dice":
                total_fg_dice / n_batches,

            "val_fg_iou":
                total_fg_iou / n_batches,
        }

        print("\nValidation")

        for k, v in metrics.items():

            print(
                f"{k}: {v:.4f}"
            )

        return metrics