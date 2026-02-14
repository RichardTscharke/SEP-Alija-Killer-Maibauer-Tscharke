import sys
import os

# Modules that are needed
from src.models.ResNetLight2 import ResNetLightCNN2
from src.train import main as train
from src.generate_csv import generate_csv
from src.explain_image import main as visualize_image
from src.explain_video import main as visualize_video
from src.demo import main as demo


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


def main():
    print("=========================================")
    print("    EMOTION RECOGNITION  -  Main Menu    ")
    print("=========================================")

    # 1 Generate CSV files
    if ask_user("Do you want to generate the CSV files from your dataset?"):
        InputFolder = get_input_folder()

        # Face Alignment needs to be put here
        #   InputFoler = ?

        print("\n--- Generating CSV Files ---")
        try:
            generate_csv(InputFolder)
        except Exception as e:
            print(f"Error during CSV generation: {e}")

    # 2 Explainable AI Image
    if ask_user("Do you want to see explainable AI visualizations for your images?"):
        InputImage = get_input_image()
        print("\n--- Starting Explainable AI Visualizations ---")
        try:
            visualize_image(InputImage)
        except Exception as e:
            print(f"Error during explainable AI visualization: {e}")

    # 3 Explainable AI Video
    if ask_user("Do you want to see explainable AI visualizations for your videos?"):
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
            demo()
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
