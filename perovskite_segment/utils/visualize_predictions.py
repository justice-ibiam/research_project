import torch
import numpy as np 
import matplotlib.pyplot as plt
from .perspective_rectification import (
    rectify_image,
    visualize_rectified,
    remove_background,
)
import cv2


def visualize_predictions(cfg,
                          model,
                          dataloader,
                          device,
                          num_images=5):
    """
    Visualize image, ground-truth mask and predicted mask.
    """

    model.eval()

    with torch.no_grad():

        count = 0

        for images, masks in dataloader:

            images = images.to(device)

            outputs = model(images)
            outputs = model(images)

            if isinstance(outputs, dict):          # DeepLabV3
                outputs = outputs["out"]

            elif isinstance(outputs, (tuple, list)):   # U²-Net
                outputs = outputs[0]

            # U-Net
            preds = (torch.sigmoid(outputs) > 0.5).float()

            batch_size = images.size(0)

            for i in range(batch_size):

                image = images[i].cpu().permute(1, 2, 0).numpy()

                gt = masks[i].cpu().squeeze().numpy()

                pred = preds[i].cpu().squeeze().numpy()

                # Perspective rectification
                foreground, mask = remove_background( image, pred ) 

                rectified = rectify_image(foreground, mask)

                visualize_rectified(cfg, image, pred, gt, rectified, num_images, count)


 