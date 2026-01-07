import os
import shutil

# Define paths for raw RAF dataset and output directory
raw_RAF_images_dir = "data/RAF_raw/Image/aligned"
label_file = "data/RAF_raw/EmoLabel/list_patition_label.txt"
output_dir = "data/RAF_processed"

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger"
}

def setup_directories():
    # Remove existing output directory if it exists
    if os.path.exists(output_dir):
        print(f"Removing existing directory: {output_dir} for a fresh start.")
        shutil.rmtree(output_dir)

    # Create necessary directories
    for split in ['train', 'test']:
        for emotion in labels.values():
            dir_path = os.path.join(output_dir, split, emotion)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Created directory: {dir_path}")

def process_data():
    
    # Check if label file exists
    if not os.path.exists(label_file):
        raise FileNotFoundError(f"Label file not found: {label_file}")
    
    print(f"starting to sort images based on labels from {label_file}")
    # Read label file
    with open(label_file, 'r') as f:
        lines = f.readlines()

    # Initialize counters
    count = 0
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
        
        #fixing filename to match aligned images
        filename_aligned = original_filename.replace('.jpg', '_aligned.jpg')

        # search for image
        image_path = os.path.join(raw_RAF_images_dir, filename_aligned)

        # Check if image exists
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}, skipping.")
            ignored_count += 1
            continue

        # Determine if image is for training or testing
        target_folder = 'train' if 'train' in filename_aligned else 'test'
        target_dir = os.path.join(output_dir, target_folder, emotion_name, filename_aligned)
        
        # Copy image to target directory
        shutil.copy(image_path, target_dir)
        count += 1

    print(f"Processed {count} images. Ignored {ignored_count} images.")

if __name__ == "__main__":
    setup_directories()
    process_data()