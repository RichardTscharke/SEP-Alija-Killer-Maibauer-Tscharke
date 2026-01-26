import os
import shutil
import random

# Define paths for raw RAF dataset (Aligned & Original)
raw_KDEF_original_dir = "data/KDEF"
#raw_KDEF_original_dir = "/Users/richardachtnull/KDEF"

#  Define paths for Output directories
output_original_dir = "data/KDEF/Image/KDEF_original_processed"
#output_original_dir = "/Users/richardachtnull/KDEF/Image/KDEF_original_processed"
output_aligned_dir = "data/KDEF/Image/KDEF_aligned_processed"
#output_aligned_dir = "/Users/richardachtnull/KDEF/Image/KDEF_aligned_processed"

# Define path for output emotion-label file
label_file = "data/KDEF/EmoLabel/list_patition_label.txt"
#label_file = "/Users/richardachtnull/KDEF/EmoLabel/list_patition_label.txt"

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

# AN=Anger, DI=Disgust, AF=Fear, HA=Happiness, SA=Sadness, SU=Surprise
CODE_TO_EMOTION_NAME = {
    "AN": "Anger",
    "DI": "Disgust",
    "AF": "Fear",
    "HA": "Happiness",
    "SA": "Sadness",
    "SU": "Surprise"
}


def sort_KDEF(filter_kdef):

    setup_directories()
    global_counter = 1

    buffers = {emotion: [] for emotion in filter_kdef.cfg}

    lf = open(label_file, "w")
    print("🔄 Starting KDEF Preparation...")

    for root, _, files in os.walk(raw_KDEF_original_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if len(file) < 7:
                continue

            emo_code = file[4:6]
            if emo_code not in CODE_TO_EMOTION_NAME:
                continue

            emotion = CODE_TO_EMOTION_NAME[emo_code]
            stem = os.path.splitext(file)[0]

            if stem.endswith("S"):
                position = "S"
            elif stem.endswith(("HL", "HR")):
                position = stem[-2:]
            else:
                continue

            if not filter_kdef.allows(emotion, position):
                continue

            buffers[emotion].append(os.path.join(root, file))

    # Sampling + Kopieren
    for emotion, paths in buffers.items():
        cfg = filter_kdef.cfg[emotion]
        random.shuffle(paths)
        selected = paths[:cfg["target"]]

        label_id = [k for k, v in labels.items() if v == emotion][0]

        for src in selected:
            new_name = f"kdef_train_{global_counter}.jpg"
            dst = os.path.join(output_original_dir, emotion, new_name)

            shutil.copy(src, dst)
            lf.write(f"{new_name} {label_id}\n")

            global_counter += 1

    lf.close()
    print("✅ prepare_KDEF finished!")
    print(f"Total images: {global_counter - 1}")


def setup_directories():
    # List of directories to setup
    target_dirs = [output_aligned_dir, output_original_dir]

    for target_dir in target_dirs:
        # Remove existing output directory if it exists
        if os.path.exists(target_dir):
            print(f"Removing existing directory: {target_dir} for a fresh start.")
            shutil.rmtree(target_dir)

        # Create necessary directories
        for emotion in labels.values():
            dir_path = os.path.join(target_dir, emotion)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

    # Create/clear the label file
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    print(f"Created label file: {label_file}")