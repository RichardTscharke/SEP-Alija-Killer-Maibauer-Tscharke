import os
import shutil
import random

# Define paths for raw RAF dataset (Aligned & Original)
raw_KDEF_original_dir = "data/KDEF"

#  Define paths for Output directories
output_original_dir = "data/KDEF/Image/KDEF_original_processed"
output_aligned_dir = "data/KDEF/Image/KDEF_aligned_processed"

# Define path for output emotion-label file
label_file = "data/KDEF/EmoLabel/list_patition_label.txt"

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

def setup_directories():
    # Delete existing output directory if exists
    if os.path.exists(output_original_dir):
        print(f"Deleting existing directory: {output_original_dir}")
        shutil.rmtree(output_original_dir)
    
    # Create output directories for each emotion
    for emotion_name in labels.values():
        path = os.path.join(output_original_dir, emotion_name)
        os.makedirs(path, exist_ok=True)
    
    # Create/clear the label file
    os.makedirs(os.path.dirname(label_file), exist_ok=True)
    print(f"Created label file: {label_file}")

# Decide whether to keep an image based on its filename and emotion according to the rules
def check_rules(filename, emotion_name):
    
    # Rules:
    # - Surprise: frontal only
    # - All others (Anger, Disgust, Fear, Sadness, Happiness): frontal + half
    # - Profile images are ignored completely
    # only 50 random images of Happiness are kept (done in main function)

    stem = os.path.splitext(filename)[0]
    # KDEF naming convention:
    # S = Straight (Frontal)
    # FL/FR = Half
    # PL/PR = Profile
    
    last_char = stem[-1]       
    last_two = stem[-2:]       

    # Determine angle
    if last_char == 'S':
        angle = "Frontal"
    elif last_two in ['FL', 'FR']:
        angle = "Half"
    else:
        return False # Profile (PL, PR) ignorieren wir komplett

    # Apply rules
    # 1. Surprise: frontal only
    if emotion_name == "Surprise":
        return angle == "Frontal"

    # 2. All others (Anger, Disgust, Fear, Sadness, Happiness): frontal + half
    return True # Keep both frontal and half for other emotions

def main():
    setup_directories()

    global_counter = 1
    processed_counter = 0

    # Buffer for Happiness images (to limit to 50)
    happy_buffer = []

    lf = open(label_file, "w")
    print("🔄 Starting KDEF Preparation...")

    for root, dirs, files in os.walk(raw_KDEF_original_dir):
        for file in files:
            if not file.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # check filename length
            if len(file) < 7: continue 
            
            # get emotion code (Index 4:6)
            emo_code = file[4:6] 
            
            # Mapping Code -> Name (z.B. "AN" -> "Anger")
            if emo_code not in CODE_TO_EMOTION_NAME:
                continue # Neutral or unknown code
            
            emotion_name = CODE_TO_EMOTION_NAME[emo_code]
            
            # check rules
            if check_rules(file, emotion_name):
                
                full_src_path = os.path.join(root, file)
                
                # special handling for Happiness
                # save to buffer first
                if emotion_name == "Happiness":
                    happy_buffer.append(full_src_path)
                
                # Normal handling for other emotions:
                else:
                    new_name = f"kdef_train_{global_counter}.jpg"
                    target_path = os.path.join(output_original_dir, emotion_name, new_name)
                    
                    # copy file to target
                    shutil.copy(full_src_path, target_path)
                    
                    # write label file
                    label_id = [k for k, v in labels.items() if v == emotion_name][0]
                    lf.write(f"{new_name} {label_id}\n")
                    
                    global_counter += 1
                    processed_counter += 1

    print(f"Prepared: {processed_counter} non-Happiness images.")

    # handle Happiness images (limit to 50)
    print(f"Found Happiness images (Front+Half): {len(happy_buffer)}")
    
    # Shuffle the buffer to get random selection
    random.shuffle(happy_buffer)
    
    # Select 50 random images
    selected_happy = happy_buffer[:50]
    print(f"Selected 50 random Happiness images.")
    
    for happy_path in selected_happy:
        new_name = f"kdef_train_{global_counter}.jpg"
        target_path = os.path.join(output_original_dir, "Happiness", new_name)
        
        shutil.copy(happy_path, target_path)
        
        # Label ID for Happiness is 4
        lf.write(f"{new_name} 4\n")
        
        global_counter += 1

    lf.close()
    print("="*30)
    print("✅ prepare_KDEF finished!")
    print(f"Total images in output folder: {global_counter - 1}")


if __name__ == "__main__":
    main()