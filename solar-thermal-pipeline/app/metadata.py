from pathlib import Path
from datetime import datetime
import json
import cv2


def generate_metadata(image_path):

    image_path = Path(image_path)

    image = cv2.imread(str(image_path))

    height, width = image.shape[:2]

    metadata = {
        "filename": image_path.name,
        "timestamp": datetime.utcnow().isoformat(),
        "sensor": "FLIR Lepton",
        "platform": "PureThermal3",
        "width": width,
        "height": height,
        "file_size_bytes": image_path.stat().st_size
    }

    output = Path("data/metadata") / f"{image_path.stem}.json"

    with open(output, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"Saved {output}")