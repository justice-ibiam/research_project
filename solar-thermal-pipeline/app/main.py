from pathlib import Path

from validator import validate_image
from metadata import generate_metadata
from segment import segment_image

image_dir = Path("data/images")

for image_path in image_dir.glob("*"):

    print(f"Validating {image_path.name}")

    validate_image(image_path)

    print("Valid image")

    generate_metadata(image_path)

    segment_image(image_path)