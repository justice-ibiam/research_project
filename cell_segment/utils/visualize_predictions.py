import torch
import numpy as np
import matplotlib.pyplot as plt


@torch.no_grad()
def visualize_predictions(
    cfg,
    dataloader,
    model,
    device,
    num_images=5,
):

    model.eval()

    count = 0

    for images, masks in dataloader:

        images = images.to(device)

        outputs = model(images)

        # Handle different model outputs

        if cfg.model.name == "deeplabv3_resnet50":

            logits = outputs["out"]

        elif isinstance(outputs, (tuple, list)):

            # U²-Net
            logits = outputs[0]

        else:

            logits = outputs

        preds = torch.argmax(
            logits,
            dim=1
        )

        batch_size = images.size(0)

        for i in range(batch_size):

            # Thermal image
            image = (
                images[i]
                .cpu()
                .permute(1, 2, 0)
                .numpy()
            )

            image = (
                image * np.array(cfg.dataset.std)
                + np.array(cfg.dataset.mean)
            )

            image = np.clip(
                image,
                0,
                1
            )

            # Ground truth

            gt = (
                masks[i]
                .cpu()
                .numpy()
            )

            # Prediction

            pred = (
                preds[i]
                .cpu()
                .numpy()
            )

            # Plot

            fig, ax = plt.subplots(
                1,
                4,
                figsize=(20, 5)
            )

            ax[0].imshow(image)

            ax[0].set_title(
                "Thermal Image"
            )

            ax[0].axis("off")

            ax[1].imshow(
                gt,
                cmap="tab20",
                vmin=0,
                vmax=12
            )

            ax[1].set_title(
                "Ground Truth"
            )

            ax[1].axis("off")

            ax[2].imshow(
                pred,
                cmap="tab20",
                vmin=0,
                vmax=12
            )

            ax[2].set_title(
                "Prediction"
            )

            ax[2].axis("off")

            ax[3].imshow(image)

            ax[3].imshow(
                pred,
                cmap="tab20",
                alpha=0.5,
                vmin=0,
                vmax=12
            )

            ax[3].set_title(
                "Prediction Overlay"
            )

            ax[3].axis("off")

            plt.tight_layout()
            plt.show()

            count += 1

            if count >= num_images:
                return