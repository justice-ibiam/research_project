from PIL import Image
import numpy as np

def trim_and_resize(image, target_size=256, threshold=10):
    # ---- Step 1: Convert to grayscale ----
    gray = np.array(image.convert("L"))

    # Rows/cols where pixels are above threshold
    rows = np.where(gray.mean(axis=1) > threshold)[0]
    cols = np.where(gray.mean(axis=0) > threshold)[0]

    # Crop bounding box
    rmin, rmax = rows[0], rows[-1]
    cmin, cmax = cols[0], cols[-1]
    cropped = image.crop((cmin, rmin, cmax, rmax))

    # Step 2: Aspect-ratio preserving resize 
    w, h = cropped.size
    scale = target_size / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cropped.resize((new_w, new_h), Image.LANCZOS)

    # Paste centered on 256x256 canvas
    canvas = Image.new("RGB", (target_size, target_size), (0, 0, 0))
    left = (target_size - new_w) // 2
    top = (target_size - new_h) // 2
    canvas.paste(resized, (left, top))

    return canvas
