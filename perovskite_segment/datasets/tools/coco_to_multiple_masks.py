from pathlib import Path
import cv2
import numpy as np
import json

ROOT = Path("/Users/justicealuu/research_project/perovskite_segment/data/cell_segmentation/valid")

JSON_FILE = ROOT / "_annotations.coco.json"

IMAGE_DIR = ROOT

MASK_DIR = ROOT / "masks"

MASK_DIR.mkdir(exist_ok=True)


with open(JSON_FILE) as f:
    coco = json.load(f)

# Build lookup table
images = {
    img["id"]: img
    for img in coco["images"]
}

# Create one empty mask for each image
masks = {}

for img in coco["images"]:

    masks[img["id"]] = np.zeros(
        (img["height"], img["width"]),
        dtype=np.uint8
    )

# Draw every annotation
for ann in coco["annotations"]:

    image_id = ann["image_id"]

    mask = masks[image_id]

    class_id = ann["category_id"]


    for polygon in ann["segmentation"]:

        pts = (
            np.array(polygon, dtype=np.float32)
            .reshape(-1, 2)
            .astype(np.int32)
        )

        cv2.fillPoly(
            mask,
            [pts],
            color=class_id
        )


# Save masks
for image_id, mask in masks.items():

    image = images[image_id]

    filename = (
        Path(image["file_name"]).stem + ".png"
    )

    cv2.imwrite(
        str(MASK_DIR / filename),
        mask
    )

print("Finished.")