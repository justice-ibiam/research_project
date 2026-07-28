import os
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
import hydra
from datasets.util import get_dataset

from datasets.solarpark.dataset import SolarPark
from models.util import get_model
from utils.perspective_rectification import (
    remove_background,
    rectify_image,
)

# from .perovskite_segment.utils.perspective_rectification import remove_background, rectify_image
@hydra.main(version_base=None, config_path='config', config_name='default.yaml')
def save_segmented_panels(cfg):

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )


    model = get_model(cfg)

    checkpoint = torch.load(
        cfg.checkpoint_path,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    split = 'valid'
    image_dir = os.path.join(cfg.dataset.path, split, "images")
    mask_dir = os.path.join(cfg.dataset.path, split, "masks")
        
    dataset = SolarPark(cfg, image_dir, mask_dir, split)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
    )

    save_dir = Path(
        "rectified_panels_test"
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    with torch.no_grad():

        for idx, (image, _) in enumerate(tqdm(dataloader)):

            image = image.to(device)

            outputs = model(image)

            if cfg.model.name == "deeplabv3_resnet50":
                pred = outputs["out"]

            elif isinstance(outputs, (tuple, list)):
                pred = outputs[0]

            else:
                pred = outputs

            pred = torch.sigmoid(pred)

            pred = (
                pred > 0.5
            ).float()

            ##################################################
            # Convert image back to numpy
            ##################################################

            img = image[0].cpu().permute(1, 2, 0).numpy()

            img = (
                img
                * np.array(cfg.dataset.std)
                + np.array(cfg.dataset.mean)
            )

            img = np.clip(
                img,
                0,
                1,
            )

            img = (
                img * 255
            ).astype(np.uint8)

            img = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR,
            )

            ##################################################
            # Convert prediction to uint8 mask
            ##################################################

            mask = (
                pred[0]
                .cpu()
                .squeeze()
                .numpy()
                * 255
            ).astype(np.uint8)

            ##################################################
            # Remove background
            ##################################################

            foreground, clean_mask = remove_background(
                img,
                mask,
            )

            ##################################################
            # Perspective rectification
            ##################################################

            rectified = rectify_image(
                foreground,
                clean_mask,
            )

            if rectified is None:
                continue

            ##################################################
            # Save image
            ##################################################

            filename = (
                save_dir
                / f"{idx:05d}.png"
            )

            cv2.imwrite(
                str(filename),
                rectified,
            )

    print("Finished!")


if __name__ == "__main__":
    save_segmented_panels()