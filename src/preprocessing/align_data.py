import cv2
from tqdm import tqdm
from pathlib import Path
from .detectors import RetinaFaceDetector
from .aligning.pipeline import preprocess_image 

def align_data(data):

    if data == "RAF":
        INPUT_DIR = Path("data/RAF_original_processed")
        OUTPUT_DIR = Path("data/RAF_aligned_processed")

        splits = [(p, p.name) for p in INPUT_DIR.iterdir() if p.is_dir()]

    elif data == "KDEF":
        INPUT_DIR = Path("data/KDEF/Image/KDEF_original_processed")
        #INPUT_DIR = Path("/Users/richardachtnull/KDEF/Image/KDEF_original_processed")
        OUTPUT_DIR = Path("data/KDEF/Image/KDEF_aligned_processed")
        #OUTPUT_DIR = Path("/Users/richardachtnull/KDEF/Image/KDEF_original_processed")

        splits = [(INPUT_DIR, "train")]


    detector = RetinaFaceDetector(device="cuda")

    success = 0
    failed = 0

    LOG_FILE = OUTPUT_DIR / "preprocess.log"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for split_dir, split in splits:
        
        out_split_dir = OUTPUT_DIR / split
        out_split_dir.mkdir(parents=True, exist_ok=True)

        for emotion_dir in sorted(split_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name
            out_emotion_dir = out_split_dir / emotion
            out_emotion_dir.mkdir(parents=True, exist_ok=True)

            images = sorted(emotion_dir.iterdir())

            for img_path in tqdm(images, desc = f"{split}/{emotion}", leave = False):

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

