from pathlib import Path
import shutil
import time

from validator import validate_image
from metadata import generate_metadata
from segment import segment_image

INCOMING = Path("data/incoming")
PROCESSED = Path("data/processed")

print("Watcher started", flush=True)

while True:

    files = list(INCOMING.glob("*"))

    if files:

        print(
            f"Detected {len(files)} new file(s)",
            flush=True
        )

    for image_path in files:

        print(
            f"Processing {image_path.name}",
            flush=True
        )

        try:

            validate_image(image_path)

            generate_metadata(image_path)

            segment_image(image_path)

            destination = (
                PROCESSED /
                image_path.name
            )

            shutil.move(
                str(image_path),
                str(destination)
            )

            print(
                f"Finished {image_path.name}",
                flush=True
            )

        except Exception as e:

            print(
                f"ERROR: {e}",
                flush=True
            )

    time.sleep(5)