import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from mtcnn import MTCNN        
from preprocessing.pipeline import preprocess_image
from preprocessing.fallback import fallback



INPUT_DIR = Path("data/RAF_raw/Image/original")
OUTPUT_DIR = Path("data/RAF_raw/Image/aligned")
LOG_FILE = OUTPUT_DIR / "preprocess.log"

def main():

    detector = MTCNN()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    log_lines = []

    for img_path in sorted(INPUT_DIR.rglob("*")):

        if img_path.suffix.lower() not in [".jpg", ".png"]:
            continue

        out_path = OUTPUT_DIR / f"{img_path.stem}_aligned{img_path.suffix}"

        image_array = np.array(Image.open(img_path))

        sample = preprocess_image(image_array, detector)

        if sample is not None:

            img = sample["image"]  # RGB, 64x64

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            continue

        crop_mtcnn, crop64 = fallback(image_array)

        sample2 = preprocess_image(crop_mtcnn, detector)

        if sample2 is not None:

            img = sample2["image"]  # RGB, 64x64

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            log_lines.append(f"{img_path.name}: fallback + second pass aligned")
            continue 

        cv2.imwrite(
        str(out_path),
        cv2.cvtColor(crop64, cv2.COLOR_RGB2BGR)
        )
        log_lines.append(f"{img_path.name}: fallback ROI only")

    # Log schreiben
    with open(LOG_FILE, "w") as f:
        f.write("\n".join(log_lines))

if __name__ == "__main__":
    main()