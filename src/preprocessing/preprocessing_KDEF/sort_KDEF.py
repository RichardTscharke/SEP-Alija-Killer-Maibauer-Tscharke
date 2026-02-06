import os
import shutil
import random
from preprocessing.sort_data import setup_directories


IMAGE_IN = "data/KDEF"

ALIGNED_OUT  = "data/KDEF/Image/KDEF_aligned_processed"
ORIGINAL_OUT = "data/KDEF/Image/KDEF_original_processed"

LABEL_OUT = "data/KDEF/EmoLabel/KDEF_labels_filtered.txt"

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

EMOTION_TO_LABEL = {v: k for k, v in labels.items()}

# Emotion in file name logic: AN=Anger, DI=Disgust, AF=Fear, HA=Happiness, SA=Sadness, SU=Surprise
CODE_TO_EMOTION_NAME = {
    "AN": "Anger",
    "DI": "Disgust",
    "AF": "Fear",
    "HA": "Happiness",
    "SA": "Sadness",
    "SU": "Surprise"
}


def sort_KDEF(filter_kdef):
    """
    Filters, samples and sorts KDEF images by emotion and head pose,
    and generates a label file compatible with the training pipeline.
    """

    #(Re)creates emotion-wise output directories for a dataset.
    #Existing directories are removed to ensure a clean preprocessing run.
    setup_directories(ALIGNED_OUT, ORIGINAL_OUT)

    # Serves for the target file names
    counter = 1

    buffers = {emotion: [] for emotion in filter_kdef.rules}

    lf = open(LABEL_OUT, "w")
    print("[INFO] Starting KDEF Preparation.")

    # Iterate over the original 35 KDEF folders
    for root, _, files in os.walk(IMAGE_IN):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            if len(file) < 7:
                continue

            # Emotion annotation is either only the 4th or the 4th+5th symbol
            emo_code = file[4:6]
            if emo_code not in CODE_TO_EMOTION_NAME:
                continue

            emotion = CODE_TO_EMOTION_NAME[emo_code]
            stem = os.path.splitext(file)[0]

            # Head position logic: "S" = Straight, "HL/HR" = Half Left/Right
            if stem.endswith("S"):
                position = "S"
            elif stem.endswith(("HL", "HR")):
                position = stem[-2:]
            # Exclude fullsided faces
            else:
                continue
            
            # Check the filter for valid positions and ratios
            if not filter_kdef.allows(emotion, position):
                continue

            buffers[emotion].append(os.path.join(root, file))

    # Sampling + Copying
    for emotion, paths in buffers.items():
        rules = filter_kdef.rules[emotion]
        random.shuffle(paths)
        selected = paths[:rules["target"]]

        label_id = EMOTION_TO_LABEL[emotion]

        for src in selected:
            new_name = f"kdef_{counter:06d}.jpg"
            dst = os.path.join(ORIGINAL_OUT, emotion, new_name)

            shutil.copy2(src, dst)
            lf.write(f"{new_name} {label_id}\n")

            counter += 1

    lf.close()
    print(f"[INFO] Sorted {counter} original images.")
    print(f"[INFO] Original images saved in: {ORIGINAL_OUT}")