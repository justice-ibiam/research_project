from pathlib import Path
import cv2


def segment_image(image_path):

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Otsu threshold
    _, mask = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    output_path = (
        Path("data/masks")
        / f"{image_path.stem}_mask.png"
    )

    cv2.imwrite(str(output_path), mask)

    print(f"Saved mask {output_path}")