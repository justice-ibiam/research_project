import torch
import matplotlib.pyplot as plt


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

                fig, ax = plt.subplots(
                    1,
                    3,
                    figsize=(15,5)
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

                plt.tight_layout()
                plt.show()

                count += 1

                if count >= num_images:
                    return