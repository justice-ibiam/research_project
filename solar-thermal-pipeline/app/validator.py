from pathlib import Path
import cv2


def validate_image(image_path):

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"{image_path} does not exist"
        )

    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(
            f"{image_path} cannot be opened"
        )

    h, w = image.shape[:2]

    if h == 0 or w == 0:
        raise ValueError(
            f"{image_path} has invalid dimensions"
        )

    return image