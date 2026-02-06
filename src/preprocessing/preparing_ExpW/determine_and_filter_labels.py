from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

LABEL_IN  = Path("data/ExpW/label/label.lst")
LABEL_OUT = Path("data/ExpW/label/label_filtered.txt")

# ExpW label logic -> RAF label logic
ExpW_TO_INTERNAL = {
    0: 6,   # Anger
    1: 3,   # Disgust
    2: 2,   # Fear
    3: 4,   # Happiness
    4: 5,   # Sadness
    5: 1,   # Surprise
    # 6 (neutral) is ignored
}
INTERNAL_TO_EMOTION = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

def determine_and_filter(ratios: dict):
    """
    Changes ExpW to single face images and downamples classes by modifying the label list file.
    Operates only on list_patition_label.txt (no image deletion).
    The label list contains the boundingbox of the final face to later crop it.
    """

    label_ratios = {
        label: ratios.get(INTERNAL_TO_EMOTION[label], 1.0)
        for label in INTERNAL_TO_EMOTION
    }
    
    # Group ExpW by image names
    grouped = defaultdict(list)

    with LABEL_IN.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            # ExpW label format: image_name face_id top left right bottom confidence label
            image_name = parts[0]
            x1 = int(parts[3])
            y1 = int(parts[2])
            x2 = int(parts[4])
            y2 = int(parts[5])
            confidence = float(parts[6])
            label = int(parts[7])

            grouped[image_name].append((confidence, x1, y1, x2, y2, label))

    print(f"[INFO] ExpW has multiple labels for one image. Unique images in label.lst: {len(grouped)}.")

    per_class = defaultdict(list)
    
    # Select the face with highest confidence per image
    for image_name in sorted(grouped):

        best = max(grouped[image_name], key=lambda x: x[0])

        confidence, x1, y1, x2, y2, original_label = best

        # Skip neutral emotions
        if original_label == 6:
            continue

        if original_label not in ExpW_TO_INTERNAL:
            continue

        internal_label = ExpW_TO_INTERNAL[original_label]

        per_class[internal_label].append((image_name, x1, y1, x2, y2))

    final_entries = []

    # Apply ratios per class
    for label, entries in per_class.items():
        ratio = label_ratios[label]
        n_total = len(entries)
        n_keep = int(n_total * ratio)

        random.shuffle(entries)
        if ratio >= 1:
            selected = entries  
        else:
            selected = entries[:n_keep]

        emotion = INTERNAL_TO_EMOTION[label]
        print(f"[INFO] ({emotion}): {len(entries)} -> {len(selected)}")

        # New ExpW label format: image_name x1 y1 x2 y2 label
        for image_name, x1, y1, x2, y2 in selected:
            final_entries.append((image_name, x1, y1, x2, y2, label))

    # Write updated label entries to label.txt
    LABEL_OUT.parent.mkdir(parents = True, exist_ok = True)

    with LABEL_OUT.open("w") as f:
        for name, x1, y1, x2, y2, label in final_entries:
            f.write(f"{name} {x1} {y1} {x2} {y2} {label}\n")

    print("[INFO] Clean up for ExpW labels finished.")
    print(f"[INFO] Final entries: {len(final_entries)}")

