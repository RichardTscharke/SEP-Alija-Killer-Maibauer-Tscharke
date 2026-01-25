import os
import shutil

# Define paths for raw RAF dataset (original)
raw_RAF_original_dir = "data/RAF_raw/Image/original"
label_file = "data/RAF_raw/EmoLabel/list_patition_label_filtered.txt"

#  Define paths for Output directories
output_aligned_dir = "data/RAF_aligned_processed"
output_original_dir = "data/RAF_original_processed"

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

def sort_RAF():
    
    setup_directories()

    # Check if label file exists
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")

    print(f"starting to sort images based on labels from {label_file}")
    
    # Read label file
    with open(label_file, "r") as f:
        lines = f.readlines()

    # Initialize counters
    count_original = 0

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
            continue

        # Determine if image is for training or testing (based on filename)
        target_folder = "train" if "train" in original_filename else "test"

        # --- PROCESS ORIGINAL IMAGES ---
        # filename remains the same for original images
        filename_original = original_filename

        # search for image in original folder
        image_path_original = os.path.join(raw_RAF_original_dir, filename_original)

        # Check if original image exists and move
        if os.path.exists(image_path_original):
            target_dir_original = os.path.join(
                output_original_dir, target_folder, emotion_name, filename_original
            )
            shutil.copy(image_path_original, target_dir_original)
            count_original += 1

    print(f"Sorted {count_original} original images.")
    print(f"Original images saved in: {output_original_dir}")
    

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
