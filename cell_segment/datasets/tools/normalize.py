import cv2
import numpy as np
from pathlib import Path


image_dir = Path("/Users/justicealuu/research_project/perovskite_segment/data/solarpark/train/images")

sum_channels = np.zeros(3, dtype=np.float64)
sum_squared = np.zeros(3, dtype=np.float64)
num_pixels = 0

image_paths = sorted(image_dir.glob("*"))

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = image.astype(np.float32) / 255.0

    pixels = image.reshape(-1, 3)

    sum_channels += pixels.sum(axis=0)
    sum_squared += (pixels ** 2).sum(axis=0)

    num_pixels += pixels.shape[0]

mean = sum_channels / num_pixels

std = np.sqrt(
    sum_squared / num_pixels - mean ** 2
)

print("Mean:", mean)
print("Std :", std)