import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from mtcnn import MTCNN        
from preprocessing.pipeline import preprocess_image
from preprocessing.fallback import fallback


def main(
        INPUT_DIR = Path("data/RAF_raw/Image/original"),
        OUTPUT_DIR = Path("data/RAF_raw/Image/aligned"),
        valid_exts = (".jpg", ".jpeg", ".png"),
        debug = False
        ):

    detector = MTCNN()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE = OUTPUT_DIR / "preprocess.log"

    log_lines = []

    for img_path in sorted(INPUT_DIR.rglob("*")):

        if img_path.suffix.lower() not in valid_exts:
            continue

        out_path = OUTPUT_DIR / f"{img_path.stem}_aligned{img_path.suffix}"

        try:
            with Image.open(img_path) as img:
                image_array = np.array(img)

        except Exception as e:
            if debug:
                raise RuntimeError(f"Failed to load image: {img_path}") from e
            else:
                with open(LOG_FILE, "w") as f:
                    f.write(f"{img_path.name}: image loading failed\n")
                continue

        sample = preprocess_image(image_array, detector)

        if sample is not None:

            img = sample["image"]  # RGB, 64x64

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            continue

        crop_mtcnn, crop64 = fallback(image_array)

        if crop_mtcnn is None or crop64 is None:
            with open(LOG_FILE, "w") as f:
                f.write(f"{img_path.name}: fallback returned invalid output\n")
            continue

        sample2 = preprocess_image(crop_mtcnn, detector)

        if sample2 is not None:

            img = sample2["image"]  # RGB, 64x64

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            )
            with open(LOG_FILE, "w") as f:
                f.write(f"{img_path.name}: fallback + second pass aligned\n")
            continue 

        cv2.imwrite(
        str(out_path),
        cv2.cvtColor(crop64, cv2.COLOR_RGB2BGR)
        )
        with open(LOG_FILE, "w") as f:
            f.write(f"{img_path.name}: fallback ROI only\n")


if __name__ == "__main__":
    main(
        INPUT_DIR = Path("data/RAF_raw/Image/original"),
        OUTPUT_DIR = Path("data/RAF_raw/Image/aligned"),
        valid_exts = (".jpg", ".jpeg", ".png"),
        debug = False
        )