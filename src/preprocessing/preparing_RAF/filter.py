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
        
        train_entries = [
            line for line in lines
            if line.startswith("train") and line.split()[1] == str(label)
        ]
        
        test_entries = [
            line for line in lines
            if line.startswith("test") and line.split()[1] == str(label)
        ]

        new_amount = min(len(train_entries), int(len(train_entries) * ratio))

        sampled_train = random.sample(train_entries, k = new_amount)

        output_lines.extend(sampled_train)
        output_lines.extend(test_entries)

        print(
            f"[INFO] ({emotion}) train: {len(train_entries)} -> {len(sampled_train)}, "
            f"test: {len(test_entries)}"
        )

    with LABEL_OUT.open("w") as f:
        for line in output_lines:
            f.write(line + "\n")
