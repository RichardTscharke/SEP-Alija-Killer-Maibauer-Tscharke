from pathlib import Path
import random

# Fixed seed for representative tests
random.seed(42)

LABEL_IN  = Path("data/RAF_raw/EmoLabel/list_patition_label.txt")
LABEL_OUT = Path("data/RAF_raw/EmoLabel/RAF_labels_filtered.txt")

RAF_LABELS = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

def filter(ratios: dict):
    """
    Downsamples RAF classes by modifying the label list file.
    Operates only on list_patition_label.txt (no image deletion).
    """

    # Read original label file
    with LABEL_IN.open("r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Define new label file
    output_lines = []

    # Loop over the original label file with ratio logic
    for label, emotion in RAF_LABELS.items():
        
        ratio = ratios.get(emotion, 1.0)

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

        # Prints the original and final amount of images per emotion class
        print(f"({emotion}): {len(entries)} -> {len(selected)}")

    # Write the final entries
    with LABEL_OUT.open("w") as f:
        for line in output_lines:
            f.write(line + "\n")
