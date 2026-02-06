import cv2
import shutil
from tqdm import tqdm
from pathlib import Path
from .detectors import RetinaFaceDetector
from .aligning.detect import detect_and_preprocess

# All our datasets have one of these extensions 
VALID_EXTS = {".jpg", ".jpeg", ".png"}

def align_data(data):
    """
    Runs face detection and alignment on emotion-wise image folders
    produced by sort_data(). Writes aligned images per emotion.
    """

    if data == "RAF":
        INPUT_DIR  = Path("data/RAF_raw/RAF_original_processed")
        OUTPUT_DIR = Path("data/RAF_raw/RAF_aligned_processed")

    elif data == "KDEF":
        INPUT_DIR  = Path("data/KDEF/Image/KDEF_original_processed")
        OUTPUT_DIR = Path("data/KDEF/Image/KDEF_aligned_processed")

    elif data == "ExpW":
        INPUT_DIR  = Path("data/ExpW/ExpW_original_processed")
        OUTPUT_DIR = Path("data/ExpW/ExpW_aligned_processed")

    else:
        raise ValueError(f"Unknown dataset: {data}")

    # Assumes GPU available
    detector = RetinaFaceDetector(device="cuda")

    # Count the successfully aligned images and fails
    success = 0
    failed = 0

    # Logs the failed alignments for debugging purposes
    LOG_FILE = OUTPUT_DIR / "preprocess.log"

    # Usually the directories are already created by sort_data()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    with LOG_FILE.open("a") as log:

        # Iterates over the 6 emotion classes
        for emotion_dir in sorted(INPUT_DIR.iterdir()):
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name
            out_emotion_dir = OUTPUT_DIR / emotion
            out_emotion_dir.mkdir(parents=True, exist_ok=True)

            images = sorted(emotion_dir.iterdir())

            # Progress visualization
            for img_path in tqdm(images, desc = f"{data}/{emotion}", leave = False):

                if img_path.suffix.lower() not in VALID_EXTS:
                    continue

                img = cv2.imread(str(img_path))

                if img is None:
                    failed += 1
                    log.write(f"{img_path}: read_failed\n")
                    continue
                
                # detect_and_preprocess returns a dict with key "image" and value aligned image or None on failure
                preprocessed = detect_and_preprocess(img, detector)

                if preprocessed is None:
                    failed += 1
                    log.write(f"{img_path}: preprocess_failed\n")
                    continue
                    
                # Name convention for aligned images
                out_path = out_emotion_dir / f"{img_path.stem}_aligned{img_path.suffix}"
                cv2.imwrite(str(out_path), preprocessed["image"])
                success += 1

    print(f"[INFO] Done. Successful: {success} | Failed: {failed}")
    print(f"[INFO] Failure log: {LOG_FILE}")

