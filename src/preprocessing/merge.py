import shutil
import random
from pathlib import Path

# Fixed seed for representative tests
random.seed(42)

OUTPUT_DIRS = {
    "train":    Path("data/train"),
    "test":     Path("data/test"),
    "validate": Path("data/validate"),
}

VALID_EXTS = {".jpg", ".jpeg", ".png"}

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


def setup_directories():
    """
    (Re)creates the global train/test/validate directories.
    Existing directories are removed for a fresh start.
    """
    for split_dir in OUTPUT_DIRS.values():
        if split_dir.exists():
            shutil.rmtree(split_dir)
        split_dir.mkdir(parents = True, exist_ok = True)

def check_directories(input_dirs):
    """
    Verifies that each dataset directory contains exactly the expected emotion classes.
    """
    for dataset_name, dataset_dir in input_dirs.items():
        found_emotions = sorted(
            [p.name for p in dataset_dir.iterdir() if p.is_dir()]
        )

        if set(found_emotions) != set(EMOTIONS):
            raise ValueError(
                f"\n[INFO] Dataset {dataset_name} has mismatching emotions!\n"
                f"[INFO] Expected: {EMOTIONS}\n"
                f"[INFO] Found:    {found_emotions}\n"
            )

        print(f"[INFO] {dataset_name} emotion folders verified.")

def merge(MERGE_SPLIT, use_aligned = True):
    """
    Merges multiple FER datasets into unified train/test/validate splits.
    Each dataset is split emotion-wise to avoid class leakage.
    Depending on use_aligned, either aligned or original images are merged.
    """

    # Select aligned or original dataset outputs as merge source
    if use_aligned:
        INPUT_DIRS = {
        "RAF":  Path("data/RAF_raw/RAF_aligned_processed"),
        "ExpW": Path("data/ExpW/ExpW_aligned_processed"),
        "KDEF": Path("data/KDEF/Image/KDEF_aligned_processed"),
        }
    else:
        INPUT_DIRS = {
        "RAF":  Path("data/RAF_raw/RAF_original_processed"),
        "ExpW": Path("data/ExpW/ExpW_original_processed"),
        "KDEF": Path("data/KDEF/Image/KDEF_original_processed"),
        }

    setup_directories()

    check_directories(INPUT_DIRS)

    for dataset_name, dataset_dir in INPUT_DIRS.items():
        ratios = MERGE_SPLIT[dataset_name]

        print(f"\n [INFO] Merging dataset: {dataset_name}")

        for emotion_dir in sorted(dataset_dir.iterdir()):
            if not emotion_dir.is_dir():
                continue

            emotion = emotion_dir.name

            images = [
                p for p in sorted(emotion_dir.iterdir())
                if p.suffix.lower() in VALID_EXTS
            ]

            if not images:
                continue

            random.shuffle(images)
            n = len(images)

            n_train = int(ratios["train"] * n)
            n_test  = int(ratios["test"] * n)
            n_val   = int(ratios["validate"] * n)

            # Due to integer rounding, the remaining samples are implicitly assigned to validation
            split_map = {
                "train":    images[:n_train],
                "test":     images[n_train:n_train + n_test],
                "validate": images[n_train + n_test:],
            }

            for split, imgs in split_map.items():
                if not imgs or ratios[split] == 0:
                    continue

                out_emotion_dir = OUTPUT_DIRS[split] /emotion
                out_emotion_dir.mkdir(parents = True, exist_ok = True)

                # Copy images into the global split directory while preserving class logic
                for img_path in imgs:
                    target_path = out_emotion_dir / img_path.name
                    shutil.copy2(img_path, target_path)

             # Per-emotion statistics help verify class balance after merging
            print(
                f"{dataset_name} | {emotion}: "
                f"train = {n_train}, test = {n_test}, validate = {n_val}"
            )