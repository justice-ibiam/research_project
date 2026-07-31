import os
import torch
import pandas as pd
from tqdm import tqdm


class ImageEvaluator:

    def __init__(
        self,
        model,
        dataloader,
        device,
        threshold=0.90,
    ):
        self.model = model
        self.dataloader = dataloader
        self.device = device
        self.threshold = threshold

    @staticmethod
    def dice_score(pred, target, eps=1e-6):

        pred = pred.float().view(-1)
        target = target.float().view(-1)

        intersection = (pred * target).sum()

        return (
            2 * intersection + eps
        ) / (
            pred.sum() + target.sum() + eps
        )

    @staticmethod
    def iou_score(pred, target, eps=1e-6):

        pred = pred.float().view(-1)
        target = target.float().view(-1)

        intersection = (pred * target).sum()

        union = pred.sum() + target.sum() - intersection

        return (
            intersection + eps
        ) / (
            union + eps
        )

    @torch.no_grad()
    def evaluate(
        self,
        save_csv="image_metrics.csv",
    ):

        self.model.eval()

        results = []

        image_index = 0

        for images, masks in tqdm(self.dataloader):

            images = images.to(self.device)

            masks = masks.to(self.device)

            outputs = self.model(images)

            if isinstance(outputs, dict):
                outputs = outputs["out"]

            elif isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            predictions = (
                torch.sigmoid(outputs) > 0.5
            ).float()

            batch_size = images.size(0)

            for b in range(batch_size):

                pred = predictions[b]

                gt = masks[b]

                dice = self.dice_score(
                    pred,
                    gt
                ).item()

                iou = self.iou_score(
                    pred,
                    gt
                ).item()

                passed = (
                    dice >= self.threshold
                    and
                    iou >= self.threshold
                )

                results.append({

                    "image":

                        image_index,

                    "dice":

                        dice,

                    "iou":

                        iou,

                    "pass":

                        passed
                })

                image_index += 1

        df = pd.DataFrame(results)

        df.to_csv(
            save_csv,
            index=False,
        )

        print()

        print("=" * 60)

        print("Per-image evaluation")

        print("=" * 60)

        print(f"Images evaluated : {len(df)}")

        print()

        print(f"Mean Dice        : {df['dice'].mean():.4f}")

        print(f"Median Dice      : {df['dice'].median():.4f}")

        print(f"Min Dice         : {df['dice'].min():.4f}")

        print(f"Max Dice         : {df['dice'].max():.4f}")

        print()

        print(f"Mean IoU         : {df['iou'].mean():.4f}")

        print(f"Median IoU       : {df['iou'].median():.4f}")

        print(f"Min IoU          : {df['iou'].min():.4f}")

        print(f"Max IoU          : {df['iou'].max():.4f}")

        passed = df["pass"].sum()

        percentage = 100 * passed / len(df)

        print()

        print(
            f"Images ≥ {self.threshold:.2f} Dice "
            f"and IoU : {passed}/{len(df)} "
            f"({percentage:.2f}%)"
        )

        print("=" * 60)

        return df