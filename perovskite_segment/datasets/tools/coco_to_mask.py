import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path("/Users/justicealuu/research_project/perovskite_segment/data/solarpark/test")

JSON_FILE = ROOT / "_annotations.coco.json"

IMAGE_DIR = ROOT

MASK_DIR = ROOT / "masks"

MASK_DIR.mkdir(exist_ok=True)


with open(JSON_FILE) as f:
    coco = json.load(f)


images = {
    img["id"]: img
    for img in coco["images"]
}


for ann in coco["annotations"]:

    image = images[ann["image_id"]]

    h = image["height"]
    w = image["width"]

    mask = np.zeros((h, w), dtype=np.uint8)

    segmentation = ann["segmentation"]

    for polygon in segmentation:

        pts = np.array(
            polygon,
            dtype=np.float32
        ).reshape(-1, 2)

        pts = pts.astype(np.int32)

        cv2.fillPoly(
            mask,
            [pts],
            255
        )

    filename = (
        Path(image["file_name"]).stem
        + ".png"
    )

    cv2.imwrite(
        str(MASK_DIR / filename),
        mask
    )

print("Finished.")