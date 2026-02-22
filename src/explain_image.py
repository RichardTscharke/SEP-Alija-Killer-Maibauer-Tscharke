import argparse
from pathlib import Path
from src.preprocessing.detectors.retinaface import RetinaFaceDetector

from src.explaining.explain_utils import get_device, resolve_model_and_layer, preprocess_image
from src.explaining.cam.explain_frame import explain_frame
from src.explaining.visualize.visualize_image import visualize

'''
This is the XAI interface (for single image) of our project.
It plots a graphic containing:
- original image
- preprocessed image
- Grad-CAM heatmap for the image
- Heatmap overlayed onto the preprocessed image
- Emotion class probablities for the image

The forward and backward hooks logic can be found within the GradCAM class.
If you trained your own model please load the model path into the 'models' folder lying within the root directory.
You can adjust the following parameters in order to achieve different results:
'''

# By default: An image of the RAF_raw dataset (assuming it is stored within the for_xai folder)
IMAGE_PATH   = Path("for_xai/xai_img.jpg")


# By default: Our best trained model
MODEL_PATH   = "model_paths/ResNetLight2_loss.pth"


# By default: A stage and layer we often achieved satisfying results with
# Useful options for ResNetLigh2 are: "stage2.conv2", "stage3.conv1" and "stage3.conv2" (stage 1 perhaps for edges)
TARGET_LAYER = "stage3"


# Determines the sufficient strength of a signal to be visible in the heatmap
THRESHOLD    = 0.4


def main(image_path):
    '''
    Runs a single-image Grad-CAM explanation pipeline:
    - loads trained model
    - resolves target convolutional layer
    - computes Grad-CAM
    - visualizes CAM overlay and class probabilities
    '''

    # Wrap the argument if it is a string
    image_path = Path(image_path)


    # Initialize device (GPU/CPU)
    device = get_device()
    print(f"[INFO] Grad-CAM inference will be calculated on device: {device}")
    
    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, device)
    print(f"[INFO] Model: {MODEL_PATH} | Target Layer: {TARGET_LAYER}")

    detector = RetinaFaceDetector(device=device)

    # open input image and retrieve metadata
    print(f"[INFO] Opening image: {image_path}")

    try:
        sample = preprocess_image(image_path, detector, device)

    except RuntimeError as e:
        print(e)
        return

    sample = explain_frame(
        sample=sample,
        model=model,
        target_layer=target_layer
    )

    # Plots a graphic of original image, aligned face, heatmap overlay and class probabilities
    visualize(
        original_img=sample["original_img"],
        aligned_img=sample["image"],
        cam_aligned=sample["cam_aligned"],
        cam_original=sample["cam_original"],
        probs=sample["probs"],
        threshold=THRESHOLD
    )


if __name__ == "__main__":
    
    # Instantiate a Argument-Parser object
    # Description is displayed for -h or --help in the terminal
    parser = argparse.ArgumentParser(description="Grad-CAM XAI for video")
    parser.add_argument(
        "image_path",
        type=str,
        nargs="?",
        default=str(IMAGE_PATH),
        help="Path to input video file"
    )

    # Read the argument from the terminal
    args = parser.parse_args()

    # Pass the video path to the main function
    main(args.image_path)



