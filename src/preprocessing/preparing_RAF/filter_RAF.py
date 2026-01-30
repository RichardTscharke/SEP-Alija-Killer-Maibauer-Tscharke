from pathlib import Path
import random

random.seed(42)

LABEL_IN = Path("data/RAF_raw/EmoLabel/list_patition_label.txt")
LABEL_OUT = Path("data/RAF_raw/EmoLabel/list_patition_label_filtered.txt")
valid_exts = (".jpg", ".jpeg", ".png")

def filter(suprise_ratio = 1,
           fear_ratio = 1,
           disgust_ratio = 1,
           happiness_ratio = 1,
           sadness_ratio = 1,
           anger_ratio = 1
            ):
    
    
    labels = {
        1: ("Surprise", suprise_ratio),
        2: ("Fear", fear_ratio),
        3: ("Disgust", disgust_ratio),
        4: ("Happiness", happiness_ratio),
        5: ("Sadness", sadness_ratio),
        6: ("Anger", anger_ratio), 
    }

    with LABEL_IN.open("r") as f:

        lines = [line.strip() for line in f if line.strip()]

    output_lines = []

    for label, (emotion, ratio) in labels.items():
        
        entries = [
            line for line in lines
            if line.split()[1] == str(label)
        ]

        if ratio >= 1:
            selected = entries
        else:
            new_amount = int(len(entries) * ratio)
            selected = random.sample(entries, k = new_amount)

        output_lines.extend(selected)

        print(f"[INFO] ({emotion}): {len(entries)} -> {len(selected)}")

    with LABEL_OUT.open("w") as f:
        for line in output_lines:
            f.write(line + "\n")
