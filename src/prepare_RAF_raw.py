import os
import shutil

# Define paths for raw RAF dataset (Aligned & Original)
#raw_RAF_aligned_dir = "data/RAF_raw/Image/aligned"
#raw_RAF_original_dir = "data/RAF_raw/Image/original"
#label_file = "data/RAF_raw/EmoLabel/list_patition_label.txt"

raw_RAF_aligned_dir = "/Users/richardachtnull/data_RICHARD/RAF_raw/Image/aligned"
raw_RAF_original_dir = "/Users/richardachtnull/data_RICHARD/RAF_raw/Image/original"
label_file = "/Users/richardachtnull/data_RICHARD/RAF_raw/EmoLabel/list_patition_label.txt"

#  Define paths for Output directories
#output_aligned_dir = "data/RAF_aligned_processed"
#output_original_dir = "data/RAF_original_processed"

output_aligned_dir = "/Users/richardachtnull/data_RICHARD/RAF_aligned_processed"
output_original_dir = "/Users/richardachtnull/data_RICHARD/RAF_original_processed"

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
    # List of directories to setup
    target_dirs = [output_aligned_dir, output_original_dir]

    for target_dir in target_dirs:
        # Remove existing output directory if it exists
        if os.path.exists(target_dir):
            print(f"Removing existing directory: {target_dir} for a fresh start.")
            shutil.rmtree(target_dir)

        # Create necessary directories
        for split in ["train", "test"]:
            for emotion in labels.values():
                dir_path = os.path.join(target_dir, split, emotion)
                os.makedirs(dir_path, exist_ok=True)
                print(f"Created directory: {dir_path}")


def process_data():

    # Check if label file exists
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    print(f"starting to sort images based on labels from {label_file}")
    # Read label file
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Initialize counters
    count_aligned = 0
    count_original = 0
    ignored_count = 0

    # Process each line in the label file
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        # Parse filename and label
        original_filename = parts[0]
        label_index = int(parts[1])

        # Remove neutral labeled images
        emotion_name = labels.get(label_index)
        if emotion_name is None:
            ignored_count += 1
            continue

        # Determine if image is for training or testing (based on filename)
        target_folder = "train" if "train" in original_filename else "test"

        # --- PROCESS ALIGNED IMAGES ---
        # fixing filename to match aligned images
        filename_aligned = original_filename.replace(".jpg", "_aligned.jpg")

        # search for image in aligned folder
        image_path_aligned = os.path.join(raw_RAF_aligned_dir, filename_aligned)

        # Check if aligned image exists and copy
        if os.path.exists(image_path_aligned):
            target_dir_aligned = os.path.join(
                output_aligned_dir, target_folder, emotion_name, filename_aligned
            )
            shutil.copy(image_path_aligned, target_dir_aligned)
            count_aligned += 1

        # --- PROCESS ORIGINAL IMAGES ---
        # filename remains the same for original images
        filename_original = original_filename

        # search for image in original folder
        image_path_original = os.path.join(raw_RAF_original_dir, filename_original)

        # Check if original image exists and copy
        if os.path.exists(image_path_original):
            target_dir_original = os.path.join(
                output_original_dir, target_folder, emotion_name, filename_original
            )
            shutil.copy(image_path_original, target_dir_original)
            count_original += 1

    print(f"Processed {count_aligned} aligned images.")
    print(f"Processed {count_original} original images.")
    print(f"Ignored {ignored_count} Images (Label: Neutral/Other).")
    print(f"Aligned images saved in: {output_aligned_dir}")
    print(f"Original images saved in: {output_original_dir}")


if __name__ == "__main__":
    setup_directories()
    process_data()
