import os
import shutil

# Define paths for raw RAF dataset (Aligned & Original)
#raw_KDEF_original_dir = "data/KDEF/Image/aligned"
raw_KDEF_original_dir = "/Users/richardachtnull/Desktop/data/KDEF"

#  Define paths for Output directories
#output_original_dir = "data/KDEF/KDEF_original_processed"
#output_aligned_dir = "data/KDEF/KDEF_aligned_processed"

output_original_dir = "/Users/richardachtnull/Desktop/data/KDEF/Image/KDEF_original_processed"
output_aligned_dir = "/Users/richardachtnull/Desktop/data/KDEF/Image/KDEF_aligned_processed"

# Define path for output emotion-label file
label_file = "/Users/richardachtnull/Desktop/data/KDEF/EmoLabel/list_patition_label.txt"

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

def setup_directories():
    target_dirs = [output_original_dir, output_aligned_dir]

    for target_dir in target_dirs:
        if os.path.exists(target_dir):
            print(f"Removing existing directory: {target_dir}")
            shutil.rmtree(target_dir)

        for emotion in labels.values():
            dir_path = os.path.join(target_dir, emotion)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")


    label_dir = os.path.dirname(label_file)
    os.makedirs(label_dir, exist_ok=True)

    with open(label_file, "w") as f:
        pass  # creates empty file

    print(f"Created label file: {label_file}")


def rename_and_move_files(fullsided = False):
    # List of possible emotion abbreviations and their corresponding emotion labels
    emos = {
        "AF": 2,
        "AN": 6,
        "DI": 3,
        "HA": 4,
        "NE": 7,
        "SA": 5,
        "SU": 1,
    }

    # List of fully blacked out images in the dataset we found manually
    missing = [551, 643, 2261, 2321, 2562, 2683, 842, 3777, 2502]

    # Keep a global counter for the image renaming
    global_counter =  1

    with open(label_file, "a") as lf:
        for root, _, files in os.walk(raw_KDEF_original_dir):

            # Ensures that no changes are done on the root level
            if root == raw_KDEF_original_dir:
                continue

            files = sorted(f for f in files if f.upper().endswith(".JPG"))

            for file in files:

                if len(file) < 7:
                    continue

                emo_code = file[4:6]  # in KDEF the emo-code is the 5th and 6th symbol

                if emo_code not in emos:
                    continue

                label = emos[emo_code]

                if label == 7:       # skipping neutral faces
                    continue

                if global_counter in missing:
                    global_counter +=  1
                    continue

                emotion_name = labels[label]

                # old name: <exactly 4 symbols that don't matter><emo><symbols that don't matter><".JPG">
                old_path = os.path.join(root,file)

                # new name: "train_<total_counter>.jpg, assuming KDEF is only used as training data"
                # special case: "train_<total_counter>_f.jpg" if human is facing one side fully
                stem = os.path.splitext(file)[0]
                last_two = stem[-2:].lower()

                if last_two in ("fl", "fr"):

                    new_name = f"train_{global_counter}_f.jpg"

                    if not fullsided:            # FLAG FULLSIDED = TRUE IF FULL SIDED FACES ARE WANTED IN THE DATA
                        global_counter +=  1
                        continue    

                else:
                    new_name = f"train_{global_counter}.jpg"

                # move to: output_original_dir/<correct emo directory determined by label>
                new_path = os.path.join(output_original_dir, emotion_name, new_name)

                shutil.move(old_path, new_path)

                # new entry in label_file: <new name> <label>
                lf.write(f"{new_name} {label}\n")

                global_counter += 1

        print(f"Processed {global_counter - 1} images.")



if __name__ == "__main__":
    setup_directories()
    rename_and_move_files(fullsided = False)   # FLAG FULLSIDED = TRUE IF FULL SIDED FACES ARE WANTED IN THE DATA

    
