from pathlib import Path
import cv2
import numpy as np
import shutil
import yaml



ROOT = Path("/Users/justicealuu/research_project/perovskite_segment/data/solarpark")

CLASS_NAMES = ["panel"]      


def mask_to_yolo(mask_path, label_path):
    """
    Converts one binary mask into a YOLOv8 segmentation label.
    """

    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if mask is None:
        print(f"Cannot read {mask_path}")
        return

    h, w = mask.shape

    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    lines = []

    for contour in contours:

        area = cv2.contourArea(contour)

        # ignore tiny blobs
        if area < 100:
            continue

        epsilon = 0.002 * cv2.arcLength(contour, True)

        contour = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        contour = contour.squeeze()

        if contour.ndim != 2:
            continue

        if len(contour) < 3:
            continue

        coords = []

        for x, y in contour:

            coords.append(x / w)
            coords.append(y / h)

        line = "0 " + " ".join(f"{c:.6f}" for c in coords)

        lines.append(line)

    with open(label_path, "w") as f:

        for line in lines:
            f.write(line + "\n")


def convert_split(split):

    image_dir = ROOT / split / "images"
    mask_dir = ROOT / split / "masks"

    label_dir = ROOT / "labels" / split
    label_dir.mkdir(parents=True, exist_ok=True)

    image_out = ROOT / "images" / split
    image_out.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*"))
    

    print(f"{split}: {len(images)} images")

    for image_path in images:

        shutil.copy(
            image_path,
            image_out / image_path.name
        )

        mask_path = None

        for ext in [".png", ".jpg", ".jpeg", ".bmp"]:
            candidate = mask_dir / (image_path.stem + ext)
            if candidate.exists():
                mask_path = candidate
                break

        if mask_path is None:
            print(f"No mask found for {image_path.name}")
            continue

        label_path = label_dir / (image_path.stem + ".txt")
        

        mask_to_yolo(
            mask_path,
            label_path
        )


def create_yaml():

    data = {
        "path": str(ROOT),
        "train": "images/train",
        "val": "images/valid",
        "test": "images/test",
        "nc": 1,
        "names": CLASS_NAMES
    }

    with open(ROOT / "dataset.yaml", "w") as f:
        yaml.dump(
            data,
            f,
            sort_keys=False
        )


if __name__ == "__main__":

    convert_split("train")
    convert_split("valid")
    convert_split("test")

    create_yaml()

    print("Finished!")