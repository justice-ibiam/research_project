import os
from tqdm import tqdm
import cv2


def apply_clahe_rgb(
    img,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
):
    """
    img: RGB image (H, W, 3), uint8
    returns: RGB image with CLAHE applied on luminance channel
    """

    # Convert RGB to LAB
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE on L channel
    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )
    l_clahe = clahe.apply(l)

    # Merge channels & convert back to RGB
    lab_clahe = cv2.merge((l_clahe, a, b))
    img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

    return img_clahe


input_dir = "/Users/justicealuu/dl-lab-25w-team07/diabetic_retinopathy/data/IDRID_dataset/processed copy/test"
output_dir = "/Users/justicealuu/dl-lab-25w-team07/diabetic_retinopathy/data/IDRID_dataset/clahe_processed/test"
os.makedirs(output_dir, exist_ok=True)

for fname in tqdm(os.listdir(input_dir)):
    if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    img = cv2.imread(os.path.join(input_dir, fname))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_clahe = apply_clahe_rgb(img)

    img_clahe = cv2.cvtColor(img_clahe, cv2.COLOR_RGB2BGR)
    cv2.imwrite(os.path.join(output_dir, fname), img_clahe)

