import os
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


@torch.no_grad()
def extract_cells(
    model,
    dataloader,
    device,
    cfg,
    output_dir="cell_results",
    padding=3,
):

    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    image_dir = os.path.join(output_dir, "cells")
    mask_dir = os.path.join(output_dir, "masks")

    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)

    statistics = []

    image_index = 0

    for images, _ in tqdm(dataloader):

        images = images.to(device)

        outputs = model(images)

        if cfg.model.name == "deeplabv3_resnet50":
            logits = outputs["out"]

        elif isinstance(outputs, (list, tuple)):
            logits = outputs[0]

        else:
            logits = outputs

        prediction = torch.argmax(logits, dim=1)

        batch_size = prediction.shape[0]

        for b in range(batch_size):

            # recover original image

            image = images[b].cpu().permute(1,2,0).numpy()

            image = (
                image * np.array(cfg.dataset.std)
                + np.array(cfg.dataset.mean)
            )

            image = np.clip(image,0,1)

            image_uint8 = (image*255).astype(np.uint8)

            pred = prediction[b].cpu().numpy()

            # process every cell

            for cell_id in range(1,13):

                cell_mask = (
                    pred == cell_id
                ).astype(np.uint8)

                if cell_mask.sum() == 0:
                    continue

                ys,xs = np.where(cell_mask)

                x1 = max(xs.min()-padding,0)
                x2 = min(xs.max()+padding,image_uint8.shape[1]-1)

                y1 = max(ys.min()-padding,0)
                y2 = min(ys.max()+padding,image_uint8.shape[0]-1)

                crop = image_uint8[
                    y1:y2+1,
                    x1:x2+1
                ]

                crop_mask = (
                    cell_mask[
                        y1:y2+1,
                        x1:x2+1
                    ]*255
                ).astype(np.uint8)

                crop = cv2.bitwise_and(
                    crop,
                    crop,
                    mask=crop_mask
                )

                # save

                filename = (
                    f"{image_index:05d}"
                    f"_cell_{cell_id:02d}"
                )

                cv2.imwrite(
                    os.path.join(
                        image_dir,
                        filename+".png"
                    ),
                    cv2.cvtColor(
                        crop,
                        cv2.COLOR_RGB2BGR
                    )
                )

                cv2.imwrite(
                    os.path.join(
                        mask_dir,
                        filename+"_mask.png"
                    ),
                    crop_mask
                )

                # statistics

                gray = cv2.cvtColor(
                    crop,
                    cv2.COLOR_RGB2GRAY
                )

                pixels = gray[crop_mask>0]

                statistics.append({

                    "image_id":
                        image_index,

                    "cell_id":
                        cell_id,

                    "x":
                        int(x1),

                    "y":
                        int(y1),

                    "width":
                        int(x2-x1+1),

                    "height":
                        int(y2-y1+1),

                    "area":
                        int(np.sum(crop_mask>0)),

                    "mean":
                        float(np.mean(pixels)),

                    "std":
                        float(np.std(pixels)),

                    "min":
                        float(np.min(pixels)),

                    "max":
                        float(np.max(pixels))
                })

            image_index += 1

    # save statistics

    df = pd.DataFrame(statistics)

    df.to_csv(
        os.path.join(
            output_dir,
            "cell_statistics.csv"
        ),
        index=False
    )

    print(
        f"Saved statistics for {len(df)} cells."
    )