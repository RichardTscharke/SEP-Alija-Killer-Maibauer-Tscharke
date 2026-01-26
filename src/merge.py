from pathlib import Path
import shutil
import random

KDEF_DIR = Path("data/KDEF/Image/KDEF_aligned_processed/train")
RAF_train_DIR = Path("data/RAF_aligned_processed/train")
RAF_test_DIR = Path("data/RAF_aligned_processed/test")
Test_Ratio_KDEF = 0.1  # Define the ratio of test set to the whole dataset
VALID_EXTS = {".jpg", ".jpeg", ".png"}


def main():

    assert KDEF_DIR.exists(), f"KDEF dir not found: {KDEF_DIR}"
    assert RAF_train_DIR.exists(), f"RAF train dir not found: {RAF_train_DIR}"
    assert RAF_test_DIR.exists(), f"RAF test dir not found: {RAF_test_DIR}"

    total_moved_train = 0
    total_moved_test = 0

    print(f"🔄 Starting Merge with {Test_Ratio_KDEF*100}% Test-Split...")

    for emotion_dir in KDEF_DIR.iterdir():

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        target_dir_train = RAF_train_DIR / emotion
        target_dir_test = RAF_test_DIR / emotion

        if not target_dir_train.exists():
            raise RuntimeError(f"Target emotion dir missing: {target_dir_train}")
        if not target_dir_test.exists():
            raise RuntimeError(f"Target emotion dir missing: {target_dir_test}")

        all_images = [
            img for img in emotion_dir.iterdir() if img.suffix.lower() in VALID_EXTS
        ]
        random.shuffle(all_images)
        split_index = int(len(all_images) * Test_Ratio_KDEF)
        test_images = all_images[:split_index]
        train_images = all_images[split_index:]

        # 1. Move test images
        for img in test_images:
            shutil.move(str(img), str(target_dir_test / img.name))
            total_moved_test += 1

        # 2. Move train images
        for img in train_images:
            shutil.move(str(img), str(target_dir_train / img.name))
            total_moved_train += 1

        print(
            f"  📂 {emotion}: {len(train_images)} -> Train | {len(test_images)} -> Test"
        )

    print("=" * 40)
    print(f"✅ Done!")
    print(f"Total Train: {total_moved_train}")
    print(f"Total Test:  {total_moved_test}")


if __name__ == "__main__":
    main()
