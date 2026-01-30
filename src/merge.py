import shutil
import random
from pathlib import Path

random.seed(42)

parameters = {
    "RAF": {
        "train":    0.7,
        "test":     0.15,
        "validate": 0.15,
    },
    "KDEF": {
        "train":    0.9,
        "test":     0.1,
        "validate": 0.0,
    },
    "ExpW": {
        "train": 0.8,
        "test": 0.1,
        "validate": 0.1,
    },
}

INPUT_DIRS = {
    "RAF":  Path("data/RAF_raw/RAF_aligned_processed"),
    "KDEF": Path("data/KDEF/Image/KDEF_aligned_processed"),
    "ExpW": Path("data/ExpW/ExpW_aligned_processed"),
}


OUTPUT_DIRS = {
    "train":    Path("data/train"),
    "test":     Path("data/test"),
    "validate": Path("data/validate"),
}

VALID_EXTS = {".jpg", ".jpeg", ".png"}

EMOTIONS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]


def setup_directories():
        for split_dir in OUTPUT_DIRS.values():
            if split_dir.exists():
                shutil.rmtree(split_dir)
            split_dir.mkdir(parents = True, exist_ok = True)

def check_directories():
    for dataset_name, dataset_dir in INPUT_DIRS.items():
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

def main():

    setup_directories()

    check_directories()

    for dataset_name, dataset_dir in INPUT_DIRS.items():
        ratios = parameters[dataset_name]

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
            n_test = int(ratios["test"] * n)
            n_val = int(ratios["validate"] * n)

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

                for img_path in imgs:
                    target_path = out_emotion_dir / img_path.name
                    shutil.copy2(img_path, target_path)

            print(
                f"[INFO] {dataset_name} | {emotion}: "
                f"train = {n_train}, test = {n_test}, validate = {n_val}"
            )

if __name__ == "__main__":
    main()