import torch

from losses import BCEDiceLoss

from metrics.dice import dice_score
from metrics.iou import iou_score
from metrics.precision_recall import precision_recall


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

        self.criterion = BCEDiceLoss()


    @torch.no_grad()
    def eval(self):

        self.model.eval()

        total_loss = 0

        total_dice = 0
        total_iou = 0
        total_precision = 0
        total_recall = 0

        n_batches = 0

        for images, masks in self.eval_loader:

            images = images.to(
                self.device
            )

            masks = masks.to(
                self.device
            )

            outputs = self.model(images)

            if self.cfg.model.name == "deeplabv3_resnet50":
            
                prediction = outputs["out"]

                loss = self.criterion(
                    prediction,
                    masks
                )
            else:            

                if isinstance(outputs, (tuple, list)):
                    # U²-Net
                    loss = torch.stack([
                        self.criterion(pred, masks)
                        for pred in outputs
                    ]).mean()
                    prediction = outputs[0]
                else:
                    prediction = outputs
                    loss = self.criterion(
                        prediction,
                        masks
                    )

            total_loss += loss.item()

            total_dice += dice_score(
                prediction,
                masks
            )

            total_iou += iou_score(
                prediction,
                masks
            )

            precision, recall = (
                precision_recall(
                    prediction,
                    masks
                )
            )

            total_precision += precision
            total_recall += recall

            n_batches += 1

        metrics = {

            "val_loss":
                total_loss / n_batches,

            "val_dice":
                total_dice / n_batches,

            "val_iou":
                total_iou / n_batches,

            "val_precision":
                total_precision / n_batches,

            "val_recall":
                total_recall / n_batches
        }

        print("\nValidation")

        for k, v in metrics.items():

            print(
                f"{k}: "
                f"{v:.4f}"
            )

        return metrics