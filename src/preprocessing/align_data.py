import cv2
from tqdm import tqdm
from pathlib import Path
from .detectors import RetinaFaceDetector
from .aligning.pipeline import preprocess_image

import onnxruntime as ort
ort.set_default_logger_severity(3)  # Errors only


def align_data(data):

    if data == "RAF":
        INPUT_DIR  = Path("data/RAF_raw/RAF_original_processed")
        OUTPUT_DIR = Path("data/RAF_raw/RAF_aligned_processed")

    elif data == "KDEF":
        INPUT_DIR  = Path("data/KDEF/Image/KDEF_original_processed")
        OUTPUT_DIR = Path("data/KDEF/Image/KDEF_aligned_processed")

    elif data == "ExpW":
        INPUT_DIR  = Path("data/ExpW/ExpW_original_processed")
        OUTPUT_DIR = Path("data/ExpW/ExpW_aligned_processed")

    detector = RetinaFaceDetector(device="cuda")

    success = 0
    failed = 0

    LOG_FILE = OUTPUT_DIR / "preprocess.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


    for emotion_dir in sorted(INPUT_DIR.iterdir()):
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        out_emotion_dir = OUTPUT_DIR / emotion
        out_emotion_dir.mkdir(parents=True, exist_ok=True)

        images = sorted(emotion_dir.iterdir())

        for img_path in tqdm(images, desc = f"{data}/{emotion}", leave = False):

            if img_path.suffix.lower() not in {".jpg", ".png", ".jpeg"}:
                continue

            img = cv2.imread(str(img_path))

            if img is None:
                failed += 1
                LOG_FILE.open("a").write(f"{img_path.name}: read_failed\n")
                continue

            preprocessed = preprocess_image(img, detector)

            if preprocessed is None:
                failed += 1
                LOG_FILE.open("a").write(f"{img_path.name}: preprocess_failed\n")
                continue
                
            out_path = out_emotion_dir / f"{img_path.stem}_aligned{img_path.suffix}"
            cv2.imwrite(str(out_path), preprocessed["image"])
            success += 1

    print(f"[INFO] Done. Successful: {success} | Failed: {failed}")
    print(f"[INFO] Failure log: {LOG_FILE}")

