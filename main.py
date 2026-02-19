import os
import cv2
import shutil
import multiprocessing as mp
from tqdm import tqdm
from pathlib import Path


# Modules that are needed
from src.preprocessing.detectors.retinaface import RetinaFaceDetector
from src.preprocessing.aligning.detect import detect_and_preprocess
from src.models.ResNetLight2 import ResNetLightCNN2
from src.train import main as train
from src.generate_csv import generate_csv
from src.explaining.explain_utils import get_device
from src.explain_image import main as visualize_image
from src.explain_video import main as visualize_video
from src.run_demo import main as demo


def ask_user(question):
    """Ask a yes/no question via input() and return their answer as True/False."""
    valid = {
        "yes": True,
        "y": True,
        "ye": True,
        "ja": True,
        "j": True,
        "no": False,
        "n": False,
        "nein": False,
    }

    while True:
        choice = input(f"{question} [y/n]: ").lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please answer with 'y' or 'n'.\n")


def get_input_folder():
    while True:
        folder = input("Please enter the path to your dataset folder: ")
        if os.path.isdir(folder):
            return folder
        else:
            print("The provided path is not a valid directory. Please try again.\n")


def get_input_image():
    while True:
        image_path = input("Please enter the path to your image file: ")
        if os.path.isfile(image_path) and image_path.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            return image_path
        else:
            print("The provided path is not a valid image file. Please try again.\n")


def get_input_video():
    while True:
        video_path = input("Please enter the path to your video file: ")
        if os.path.isfile(video_path) and video_path.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            return video_path
        else:
            print("The provided path is not a valid video file. Please try again.\n")


def align(InputFolder: Path):
    
    TargetFolder = InputFolder.parent / f"{InputFolder.name}_aligned"

    # Delete target folder for a fresh start
    if TargetFolder.exists():
        shutil.rmtree(TargetFolder)

    # Create new target folder
    TargetFolder.mkdir(parents=True, exist_ok=True)

    print(f"Target folder for aligned images initialized at: {TargetFolder}")

    # Valid image extensions
    valid_exts = {".jpg", ".jpeg", ".png"}

    # Logs the failed alignments for debugging purposes
    log_file = InputFolder.parent / "aligning.log"

   # Count the successfully aligned images and fails
    success = 0
    failed = 0

    # Chooses cuda > mps > cpu based on availability
    detector = RetinaFaceDetector(device=get_device())

    # Logging
    with log_file.open("w") as log:
            
        # Progress visualization
        for img_path in tqdm(InputFolder.rglob("*"), leave=False):

            if img_path.suffix.lower() not in valid_exts:
                log.write(f"{img_path}: invalid_extension\n")
                continue
            
            # Load image
            img = cv2.imread(str(img_path))

            if img is None:
                failed += 1
                log.write(f"{img_path}: read_failed\n")
                continue
                
            # detect_and_preprocess returns a dict with key "image" and value aligned image or None on failure
            preprocessed = detect_and_preprocess(img, detector)

            if preprocessed is None:
                failed += 1
                log.write(f"{img_path}: preprocess_failed\n")
                continue
                    
            # Name convention for aligned images
            out_path = TargetFolder / f"{img_path.stem}_aligned{img_path.suffix}"
            cv2.imwrite(str(out_path), preprocessed["image"])
            success += 1

    print(f"Alignment succeeded. Successful: {success} | Failed: {failed}")
    print(f"Find the failure log at: {log_file}")

    return TargetFolder


def main():
    print("=========================================")
    print("    EMOTION RECOGNITION  -  Main Menu    ")
    print("=========================================")

    # 1 Generate CSV files
    if ask_user("Do you want to generate the CSV files from your dataset?"):
        InputFolder = Path(get_input_folder())
        TargetFolder = InputFolder

        try:
            TargetFolder = align(InputFolder)

        except Exception as e:
            print(f"Error during aligning: {e}")
            print(f"Falling back to original dataset at {InputFolder}")

        print("\n--- Generating CSV Files ---")
        try:
            generate_csv(TargetFolder)
        except Exception as e:
            print(f"Error during CSV generation: {e}")

    # 2 Explainable AI Image
    if ask_user("Do you want to see explainable AI visualizations for one images?"):
        InputImage = get_input_image()
        print("\n--- Starting Explainable AI Visualizations ---")
        try:
            #visualize_image(InputImage)
            p = mp.Process(
                target=visualize_image,
                args=(InputImage,)
            )
            p.start()
            p.join()
        except Exception as e:
            print(f"Error during explainable AI visualization: {e}")

    # 3 Explainable AI Video
    if ask_user("Do you want to see explainable AI visualizations for a video?"):
        InputVideo = get_input_video()
        print("\n--- Starting Explainable AI Video Visualizations ---")
        try:
            visualize_video(InputVideo)
        except Exception as e:
            print(f"Error during explainable AI video visualization: {e}")

    # 4 Demo
    if ask_user("Do you want to run a demo on your webcam input?"):
        print("\n--- Starting Demo ---")
        try:
            p = mp.Process(
                target=demo,
            )
            p.start()
            p.join()
            #demo()
        except Exception as e:
            print(f"Error during demo: {e}")

    # 5. Training
    if ask_user("Do you want to retrain the model on your own data?"):
        if ask_user(
            "Did you make sure the Dataset is placed as described in the Readme?"
        ):
            print("\n--- Starting Training ---")

            try:
                train()
            except Exception as e:
                print(f"Error during training: {e}")
        else:
            print(
                "Please make sure your dataset is correctly placed before starting training. Refer to the Readme for instructions."
            )
            print("Training aborted.")

    print("\nThank you for using the Emotion Recognition System! Goodbye!")


if __name__ == "__main__":

    mp.set_start_method("spawn", force=True)

    main()