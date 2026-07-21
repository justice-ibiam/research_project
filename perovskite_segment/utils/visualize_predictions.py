import torch
import numpy as np 
import matplotlib.pyplot as plt
from utils.perspective_rectification import find_panel_contour, get_panel_corners, order_points, rectify_image, remove_background


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
                # pred = (pred * 255).astype(np.uint8)
                contour = find_panel_contour(mask)
                corners= get_panel_corners(contour)
                corners = order_points(corners)
                rectified = rectify_image(foreground, corners)



                fig, ax = plt.subplots(
                    1,
                    4,
                    figsize=(16,4)
                )

                ax[0].imshow(image)
                ax[0].set_title("Thermal Image")
                ax[0].axis("off")

                ax[1].imshow(gt, cmap="gray")
                ax[1].set_title("Ground Truth")
                ax[1].axis("off")

                ax[2].imshow(pred, cmap="gray")
                ax[2].set_title("Prediction")
                ax[2].axis("off")

                print(type(rectified))

                if rectified is not None:
                    print(rectified.dtype)
                    print(rectified.shape)

                ax[3].imshow(rectified, cmap="gray")
                ax[3].set_title("Rectified")
                ax[3].axis("off")

                plt.tight_layout()
                plt.show()

                count += 1

                if count >= num_images:
                    return