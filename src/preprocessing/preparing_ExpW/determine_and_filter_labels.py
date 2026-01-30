from pathlib import Path
from collections import defaultdict
import random

random.seed(42)

#LABEL_IN = Path("data/ExpW/label/label.lst")
LABEL_IN = Path("/Users/richardachtnull/Desktop/ExpW/label/label.lst")
#LABEL_OUT = Path("data/ExpW/label/label.txt")
LABEL_OUT = Path("/Users/richardachtnull/Desktop/ExpW/label/label.txt")

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

def determine_and_filter(surprise_ratio  = 1,
                         fear_ratio      = 1,
                         disgust_ratio   = 1,
                         happiness_ratio = 1,
                         sadness_ratio   = 1,
                         anger_ratio     = 1):
    
    ratios = {
        1: surprise_ratio,
        2: fear_ratio,
        3: disgust_ratio,
        4: happiness_ratio,
        5: sadness_ratio,
        6: anger_ratio,
    }

    # 1. Group ExpW by image names
    grouped = defaultdict(list)

    with LABEL_IN.open("r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8:
                continue

            image_name = parts[0]
            x1 = int(parts[3])
            y1 = int(parts[2])
            x2 = int(parts[4])
            y2 = int(parts[5])
            confidence = float(parts[6])
            label = int(parts[7])

            grouped[image_name].append((confidence, x1, y1, x2, y2, label))

    print(f"[INFO] Unique images in label.lst: {len(grouped)}.")

    # 2 Select the face with highest confidence per image
    per_class = defaultdict(list)
    
    for image_name in sorted(grouped):

        best = max(grouped[image_name], key=lambda x: x[0])

        confidence, x1, y1, x2, y2, original_label = best

        # skip neutral
        if original_label == 6:
            continue

        if original_label not in ExpW_TO_INTERNAL:
            continue

        internal_label = ExpW_TO_INTERNAL[original_label]

        per_class[internal_label].append((image_name, x1, y1, x2, y2))

    # 3. Apply ratios per class
    final_entries = []
    class_counter = {}

    for label, entries in per_class.items():
        ratio = ratios[label]
        n_total = len(entries)
        n_keep = int(n_total * ratio)

        random.shuffle(entries)
        selected = entries[:n_keep] if ratio < 1 else entries

        emotion = INTERNAL_TO_EMOTION[label]
        print(f"[INFO] ({emotion}): {len(entries)} -> {len(selected)}")

        class_counter[label] = len(selected)

        for image_name, x1, y1, x2, y2 in selected:
            final_entries.append((image_name, x1, y1, x2, y2, label))

    # 4. Write updated label entries to label.txt
    LABEL_OUT.parent.mkdir(parents = True, exist_ok = True)

    with LABEL_OUT.open("w") as f:
        for name, x1, y1, x2, y2, label in final_entries:
            f.write(f"{name} {x1} {y1} {x2} {y2} {label}\n")

    print("[INFO] Clean up for ExpW labels finished.")
    print(f"[INFO] Final entries: {len(final_entries)}")
    print("[INFO] Images per emotion class:")
