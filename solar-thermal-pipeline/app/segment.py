from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import cv2
import torch
import os
import hydra
from albumentations.pytorch import ToTensorV2

# from perovskite_segment.models.util import get_model
from perovskite_segment.utils.perspective_rectification import rectify_image, remove_background

from visualize_rectified import visualize_rectified
import albumentations as A
import numpy as np
import pandas as pd

@hydra.main(
    version_base=None,
    config_path="config",
    config_name="default.yaml"
)
def main(cfg):
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    from perovskite_segment.models.util import get_model
    model = get_model(cfg)

    checkpoint = torch.load(
        cfg.checkpoint_path_panel,
        map_location=device,
        weights_only=False
    )

    save_dir = Path(
            "rectified_panels_test"
        )
    
    save_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()



    image_path = Path(cfg.image_path)

    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )
    image = cv2.resize(
            image,
            (320,240),
            interpolation=cv2.INTER_LINEAR
        )

    # keep a copy for visualization
    original_image = image.copy()

    # Same preprocessing used during training

    transform = A.Compose([
        A.Normalize(
            mean=cfg.dataset.mean,
            std=cfg.dataset.std,
        ),
        ToTensorV2(),
    ])

    image = transform(image=image)["image"]

    image = image.unsqueeze(0)
    

    with torch.no_grad():
        image = image.to(device)

        output = model(image)

        if cfg.model.name == "deeplabv3_resnet50":
            pred = output["out"]

        elif isinstance(output, (tuple, list)):
            pred = output[0]

        else:
            pred = output

        pred = torch.sigmoid(pred)

        pred = (
            pred > 0.5
        ).float()

        # Convert image back to numpy

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

        # Convert prediction to uint8 mask

        mask = (
            pred[0]
            .cpu()
            .squeeze()
            .numpy()
            * 255
        ).astype(np.uint8)

        # Remove background

        foreground, clean_mask = remove_background(
            img,
            mask,
        )

        # Perspective rectification

        rectified = rectify_image(
            foreground,
            clean_mask,
        )


        # Save image

    #     filename = (
    #         save_dir
    #         / f"testing.png"
    #     )

    #     cv2.imwrite(
    #         str(filename),
    #         rectified,
    #     )

    # print("Finished!")


    #################################
    # Start of second model#
    from cell_segment.models.util import get_model
    model = get_model(
        cfg
    )

    if os.path.exists(
        cfg.checkpoint_path_cell
    ):

        checkpoint = torch.load(
            cfg.checkpoint_path_cell,
            map_location=device,
            weights_only=False
        )
        

        model.load_state_dict(
            checkpoint["model_state"]
        )
        model.to(device)
    image = rectified
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )
    image = cv2.resize(
        image,
        (320,240),
        interpolation=cv2.INTER_LINEAR
    )
    image = transform(image=image)["image"]

    image = image.unsqueeze(0)
    model.eval()

    output_dir="cell_results_test"
    padding=3

    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, "cells")
    mask_dir = os.path.join(output_dir, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    statistics = []

    image_index = 0


    image = image.to(device)

    output = model(image)

    if cfg.model.name == "deeplabv3_resnet50":
        logits = output["out"]

    elif isinstance(output, (list, tuple)):
        logits = output[0]

    else:
        logits = output

    prediction = torch.argmax(logits, dim=1)





    # recover original image
    print(image.shape)
    image = image.squeeze(0)
    print(image.shape)
    image = image.cpu().permute(1,2,0).numpy()

    image = (
        image * np.array(cfg.dataset.std)
        + np.array(cfg.dataset.mean)
    )

    image = np.clip(image,0,1)

    image_uint8 = (image*255).astype(np.uint8)

    pred = prediction.squeeze(0).cpu().numpy().astype(np.uint8)

    print("pred.shape:", pred.shape)
    print("labels:", np.unique(pred))

    # process every cell
    print(f"pred.shape: {pred.shape}")
    H, W = pred.shape

    for cell_id in range(1, 13):

        print(f"\nProcessing Cell {cell_id}")

        # Binary mask for this cell
        cell_mask = (pred == cell_id).astype(np.uint8)

        if cell_mask.sum() == 0:
            print("No pixels.")
            continue

        ys, xs = np.where(cell_mask)

        if len(xs) == 0:
            continue

        # Bounding box
        x1 = max(0, xs.min() - padding)
        x2 = min(W, xs.max() + padding + 1)

        y1 = max(0, ys.min() - padding)
        y2 = min(H, ys.max() + padding + 1)

        crop = image_uint8[y1:y2, x1:x2].copy()

        crop_mask = (
            cell_mask[y1:y2, x1:x2] * 255
        ).astype(np.uint8)

        if crop.size == 0:
            print("Empty crop.")
            continue

        crop = cv2.bitwise_and(
            crop,
            crop,
            mask=crop_mask
        )

        filename = f"{image_index:05d}_cell_{cell_id:02d}"

        rgb_path = os.path.join(
            image_dir,
            filename + ".png"
        )

        mask_path = os.path.join(
            mask_dir,
            filename + "_mask.png"
        )

        cv2.imwrite(
            rgb_path,
            cv2.cvtColor(
                crop,
                cv2.COLOR_RGB2BGR
            )
        )

        cv2.imwrite(
            mask_path,
            crop_mask
        )

        # ------------------------
        # Statistics
        # ------------------------

        gray = cv2.cvtColor(
            crop,
            cv2.COLOR_RGB2GRAY
        )

        pixels = gray[crop_mask > 0]

        if len(pixels) == 0:
            continue

        statistics.append({

            "image_id": image_index,

            "cell_id": cell_id,

            "x": int(x1),

            "y": int(y1),

            "width": int(x2 - x1),

            "height": int(y2 - y1),

            "area": int(len(pixels)),

            "mean": float(np.mean(pixels)),

            "std": float(np.std(pixels)),

            "min": float(np.min(pixels)),

            "max": float(np.max(pixels))
        })
        save_statistics(statistics, output_dir)


def save_statistics(
    statistics,
    output_dir
):

    df = pd.DataFrame(statistics)

    csv_path = os.path.join(
        output_dir,
        "cell_statistics_test.csv"
    )

    df.to_csv(
        csv_path,
        index=False
    )

    print(f"\nSaved {len(df)} cell statistics.")

   

if __name__ == "__main__":
    main()