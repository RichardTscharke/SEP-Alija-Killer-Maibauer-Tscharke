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
    """
    Creates emotion-wise folders from dataset-specific label files
    ALIGNED_OUT is prepared here but populated later by align_data()
    For ExpW: performs face cropping based on bounding boxes since multifaced images exist
    """

    if data == "RAF":
        IMAGE_IN = "data/RAF/Image/original"
        LABEL_IN = "data/RAF/EmoLabel/RAF_labels_filtered.txt"

        ALIGNED_OUT  = "data/RAF/RAF_aligned_processed"
        ORIGINAL_OUT = "data/RAF/RAF_original_processed"

    elif data == "ExpW":
        #IMAGE_IN = "data/ExpW/image/origin"
        IMAGE_IN = "data/ExpW/image"
        LABEL_IN = "data/ExpW/label/ExpW_labels_filtered.txt"

        ALIGNED_OUT  = "data/ExpW/ExpW_aligned_processed"
        ORIGINAL_OUT = "data/ExpW/ExpW_original_processed"

    else:
        raise ValueError(f"Unknown dataset: {data}")

    # Serves for the target file names
    counter = 0
    
    setup_directories(ALIGNED_OUT, ORIGINAL_OUT)

    # Check if label file exists
    if not os.path.exists(LABEL_IN):
        raise FileNotFoundError(f"Label file not found: {LABEL_IN}")

    print(f"[INFO] Sorting images based on labels from {LABEL_IN}...")
    
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
            # ExpW label format: image_name face_id top left right bottom confidence label
            label_index = int(parts[5])
            x1, y1, x2, y2 = map(int, parts[1:5])

        # Identify label <-> emotion logic
        emotion_name = labels.get(label_index)
        # Remove neutral labeled images
        if emotion_name is None:
            continue

        # Filename remains the same for original images
        filename_original = original_filename

        # Search for image in original folder
        image_path_original = os.path.join(IMAGE_IN, filename_original)

        # Check if original image exists and move
        if os.path.exists(image_path_original):
            target_dir_original = os.path.join(
                ORIGINAL_OUT, emotion_name
            )
            os.makedirs(target_dir_original, exist_ok=True)

            counter += 1

            # Create dataset specific filenames
            if data == "RAF":
                new_name = f"raf_{counter:06d}.jpg"
            elif data == "ExpW":
                new_name = f"expW_{counter:06d}.jpg"

            target_path = os.path.join(target_dir_original, new_name)
            
            # Copy data for quick changes in data distribution for new setups
            if data == "RAF":
                shutil.copy2(image_path_original, target_path)

            elif data == "ExpW":
                img = cv2.imread(image_path_original)
                if img is None:
                    continue

                # Eliminate other faces so that the face detector finds the same selected face only
                face = crop_face(img, x1, y1, x2, y2)

                if face is None or face.size == 0:
                    continue
                cv2.imwrite(target_path, face)

    if data == "RAF":
        print(f"[INFO] Sorted {counter} original images.")
    elif data == "ExpW":
        print(f"[INFO] Sorted and cropped {counter} original images.")
    print(f"[INFO] Original images saved in: {ORIGINAL_OUT}")
    

def setup_directories(ALIGNED_OUT, ORIGINAL_OUT):
    """
    (Re)creates emotion-wise output directories for a dataset.
    Existing directories are removed to ensure a clean preprocessing run.
    """

    target_dirs = [ALIGNED_OUT, ORIGINAL_OUT]

    for target_dir in target_dirs:
        # Remove existing output directory if it exists
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir)

        # Create necessary directories
        for emotion in labels.values():
            dir_path = os.path.join(target_dir, emotion)
            os.makedirs(dir_path, exist_ok=True)

    print(f"[INFO] Directories set up sucessfully.")


def crop_face(img, x1, y1, x2, y2, scale = 4.0):
    """
    Crops and enlarges a face bounding box from an image given by the dataset.
    The box is scaled around its center by the given factor and clipped
    to image boundaries.
    """

    h_img, w_img = img.shape[:2]

    if x2 <= x1 or y2 <= y1:
        return None

    w = x2 - x1
    h = y2 - y1

    # RetinaFace struggles to detect really big and close faces
    if w < 10 or h < 10:
        return None
    
    # Crop around the center
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




