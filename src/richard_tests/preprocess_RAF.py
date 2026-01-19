import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from mtcnn import MTCNN        
from preprocessing.pipeline import preprocess_image
from preprocessing.fallback import fallback
from collections import Counter


def main(
        INPUT_DIR = Path("data/RAF_raw/Image/original"),
        OUTPUT_DIR = Path("data/RAF_raw/Image/aligned"),
        valid_exts = (".jpg", ".jpeg", ".png"),
        debug = False
        ):

    detector = MTCNN()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    LOG_FILE = OUTPUT_DIR / "preprocess.log"

    stats = Counter()

    for emotion_dir in sorted(INPUT_DIR.iterdir()):

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        out_emotion_dir = OUTPUT_DIR / emotion
        out_emotion_dir.mkdir(parents = True, exist_ok = True)

        for img_path in sorted(emotion_dir.iterdir()):

            if img_path.suffix.lower() not in valid_exts:
                continue

            try:
                with Image.open(img_path) as img:
                    image_array = np.array(img)

            except Exception:
                with open(LOG_FILE, "a") as f:
                    f.write(f"{img_path}: image loading failed\n")

                if debug:
                    raise

                continue 

            aligned_img, status = call_preprocess_image(image_array, detector, debug = debug)

            stats[status] += 1

            if status != "mtcnn":
                with open(LOG_FILE, "a") as f:
                    f.write(f"{img_path}: {status}\n")

            if aligned_img is None:
                continue

            out_name = f"{img_path.stem}_aligned{img_path.suffix}"
            out_path = out_emotion_dir / out_name

            cv2.imwrite(str(out_path), cv2.cvtColor(aligned_img, cv2.COLOR_RGB2BGR))


    print("Alignment statistics:")
    for k, v in stats.items():
        print(f"{k}: {v}")



def call_preprocess_image(image_array, detector, debug = False):

    sample = preprocess_image(image_array, detector, debug = debug)

    if sample is not None:
        return sample["image"], "mtcnn"

    crop_mtcnn, crop64 = fallback(image_array)

    if crop_mtcnn is None or crop64 is None:
        return None, "fallback_failed"

    sample_fallback = preprocess_image(crop_mtcnn, detector)

    if sample_fallback is not None:
        return sample_fallback["image"], "fallback+mtcnn"

    return crop64, "fallback+mtcnn_failed+ROI"


if __name__ == "__main__":
    main(
        INPUT_DIR = Path("/Users/richardachtnull/Desktop/data/KDEF/Image/KDEF_original_processed"),
        OUTPUT_DIR = Path("/Users/richardachtnull/Desktop/data/KDEF/Image/KDEF_aligned_processed"),
        valid_exts = (".jpg", ".jpeg", ".png"),
        debug = False
        )