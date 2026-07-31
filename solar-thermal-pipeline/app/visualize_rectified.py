import numpy as np
import matplotlib.pyplot as plt

def visualize_rectified( cfg, image, pred, rectified):

    fig, ax = plt.subplots(
        1,
        4,
        figsize=(16,4)
    )


    image = image * np.array(cfg.dataset.std) + np.array(cfg.dataset.mean)
    image = np.clip(image, 0, 1)
    ax[0].imshow(image)
    ax[0].set_title("Thermal Image")
    ax[0].axis("off")

    ax[1].imshow(pred, cmap="gray")
    ax[1].set_title("Prediction")
    ax[1].axis("off")


    rgb = rectified[:, :, :3]
    alpha = rectified[:, :, 3:] / 255.0

    rgb = rgb * np.array(cfg.dataset.std) + np.array(cfg.dataset.mean)
    rgb = np.clip(rgb, 0, 1)

    rectified = np.concatenate([rgb, alpha], axis=2)
    ax[2].imshow(rectified, cmap="gray")
    ax[2].set_title("Rectified")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()