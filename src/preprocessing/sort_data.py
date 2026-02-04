import os
import cv2
import shutil

# Define Emotion labels
labels = {
    1: "Surprise",
    2: "Fear",
    3: "Disgust",
    4: "Happiness",
    5: "Sadness",
    6: "Anger",
}

def sort_data(data):

    if data == "RAF":
        IMAGE_IN = "data/RAF_raw/Image/original"
        LABEL_IN = "data/RAF_raw/EmoLabel/list_patition_label_filtered.txt"

        ALIGNED_OUT  = "data/RAF_raw/RAF_aligned_processed"
        ORIGINAL_OUT = "data/RAF_raw/RAF_original_processed"

    elif data == "ExpW":
        IMAGE_IN = "data/ExpW/image"
        LABEL_IN = "data/ExpW/label/label.txt"

        ALIGNED_OUT  = "data/ExpW/ExpW_aligned_processed"
        ORIGINAL_OUT = "data/ExpW/ExpW_original_processed"

    else:
        raise ValueError(f"Unknown dataset: {data}")

    raf_counter  = 0
    expW_counter = 0
    
    setup_directories(ALIGNED_OUT, ORIGINAL_OUT)

    # Check if label file exists
    if not os.path.exists(LABEL_IN):
        raise FileNotFoundError(f"Label file not found: {LABEL_IN}")

    print(f"starting to sort images based on labels from {LABEL_IN}")
    
    # Read label file
    with open(LABEL_IN, "r") as f:
        lines = f.readlines()

    # Process each line in the label file
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue

        # Parse filename and label
        original_filename = parts[0]
        if data == "RAF":
            label_index = int(parts[1])
        elif data == "ExpW":
            label_index = int(parts[5])
            x1, y1, x2, y2 = map(int, parts[1:5])

        # Remove neutral labeled images
        emotion_name = labels.get(label_index)
        if emotion_name is None:
            continue

        # --- PROCESS ORIGINAL IMAGES ---
        # filename remains the same for original images
        filename_original = original_filename

        # search for image in original folder
        image_path_original = os.path.join(IMAGE_IN, filename_original)

        # Check if original image exists and move
        if os.path.exists(image_path_original):
            target_dir_original = os.path.join(
                ORIGINAL_OUT, emotion_name
            )
            os.makedirs(target_dir_original, exist_ok=True)

            if data == "RAF":
                raf_counter += 1
                new_name = f"raf_{raf_counter:06d}.jpg"
            elif data == "ExpW":
                expW_counter += 1
                new_name = f"expW_{expW_counter:06d}.jpg"

            target_path = os.path.join(target_dir_original, new_name)
            
            if data == "RAF":
                shutil.copy2(image_path_original, target_path)

            elif data == "ExpW":
                img = cv2.imread(image_path_original)
                if img is None:
                    continue
                face = crop_face(img, x1, y1, x2, y2)
                if face is None or face.size == 0:
                    continue
                if face.size == 0:
                    continue
                cv2.imwrite(target_path, face)

    if data == "RAF":
        print(f"Sorted {raf_counter} original images.")
    elif data == "ExpW":
        print(f"Sorted and cropped {expW_counter} original images.")
    print(f"Original images saved in: {ORIGINAL_OUT}")
    

def setup_directories(ALIGNED_OUT, ORIGINAL_OUT):
    # List of directories to setup
    target_dirs = [ALIGNED_OUT, ORIGINAL_OUT]

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


def crop_face(img, x1, y1, x2, y2, scale = 3.0):
    h_img, w_img = img.shape[:2]

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1

    if w < 10 or h < 10:
        return None

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    new_w = int(w * scale)
    new_h = int(h * scale)

    nx1 = max(0, cx - new_w // 2)
    ny1 = max(0, cy - new_h // 2)
    nx2 = min(w_img, cx + new_w // 2)
    ny2 = min(h_img, cy + new_h // 2)

    if nx2 <= nx1 or ny2 <= ny1:
        return None

    return img[ny1:ny2, nx1:nx2]




