from pathlib import Path
import shutil

KDEF_DIR = Path("data/KDEF/Image/KDEF_aligned_processed")
RAF_DIR  = Path("data/RAF_aligned_processed/train")

VALID_EXTS = {".jpg", ".jpeg", ".png"}

def main():

    assert KDEF_DIR.exists(), f"KDEF dir not found: {KDEF_DIR}"
    assert RAF_DIR.exists(), f"RAF dir not found: {RAF_DIR}"

    total = 0

    for emotion_dir in KDEF_DIR.iterdir():

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name
        target_dir = RAF_DIR / emotion

        if not target_dir.exists():
            raise RuntimeError(f"Target emotion dir missing: {target_dir}")

        moved = 0

        for img in emotion_dir.iterdir():

            if img.suffix.lower() not in VALID_EXTS:
                continue

            target_path = target_dir / img.name

            if target_path.exists():
                raise RuntimeError(f"File collision: {target_path}")

            shutil.move(str(img), str(target_path))
            moved += 1

        print(f"{emotion}: moved {moved} images")
        total += moved

    print(f"\nTotal images moved: {total}")


if __name__ == "__main__":
    main()
