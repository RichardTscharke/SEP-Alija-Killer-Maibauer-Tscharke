import argparse
from tqdm import tqdm
from pathlib import Path

from src.preprocessing.detectors.retinaface import RetinaFaceDetector
from src.explaining.process_frame import process_frame
from src.explaining.explain_utils import get_device, resolve_model_and_layer
from src.explaining.video_utils import open_video, create_video_writer
from src.explaining.visualize.visualize_video.cam_smoother import CamSmoother
from src.explaining.visualize.visualize_video.label_smoother import LabelSmoother
from src.explaining.visualize.visualize_video.label_stabilizer import LabelStabilizer


VIDEO_PATH = Path("data/xai_vid.MOV")

MODEL_PATH = Path("model_paths/ResNetLight2_loss.pth")

TARGET_LAYER = "stage3"

# Refers to the minimum confidence a prediction must achieve to be displayed in the label
MIN_CONF = 0.3

# Refers to the minimum signal strength regions must achieve in order to be displayed in the heatmap
THRESHOLD = 0.4


def main(video_path):
    '''
    Runs Grad-CAM based emotion explanation on a video file.
    Takes the input path of the video and prints the path to the stored output video.
    Pipeline:
    - Load model and target layer
    - Initialize retina face detector and smoothing utilities
    - Process video frame-by-frame with Grad-CAM overlays and labels
    - Write annotated frames to output video
    '''

    # Wrap the argument if it is a string
    video_path = Path(video_path)

    # Initialize device (GPU/CPU)
    device = get_device()
    print(f"[INFO] Grad-CAM inference will be calculated on device: {device}")
    
    # Define the output path
    output_path = video_path.with_name(f"{video_path.stem}_explained{video_path.suffix}")

    # Load model and resolve target convolutional layer for Grad-CAM
    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, device)
    print(f"[INFO] Model: {MODEL_PATH} | Target Layer: {TARGET_LAYER}")

    # Initialize face detector
    detector = RetinaFaceDetector(device=device)

    # open input video and retrieve metadata
    print(f"[INFO] Opening video: {video_path}")
    cap, fps, total_frames = open_video(video_path)
    print(f"[INFO] Video FPS (source): {fps:.2f} | Total Frames: {total_frames}")

    # Temporal smoothing for CAMs to reduce heatmap flickering
    cam_smoother = CamSmoother(alpha=0.2, every_nth_frame=1)

    # Temporal smoothing and stabilization for emotion label overlay
    label_smoother = LabelSmoother(alpha=0.3, every_nth_frame=5)

    # min_conf refers to the minimum confidence a prediction must achieve to be written out
    # In our case we show the top 2 classes who achieved this required confidence
    label_stabilizer = LabelStabilizer(min_conf=MIN_CONF)

    writer = None

    # Frame per frame processing
    for frame_idx in tqdm(range(total_frames), desc="Explaining video", unit="frame"):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Full XAI pipeline for a single frame (detect -> CAM -> overlay -> label)
        overlayed = process_frame(
            frame,
            detector,
            model,
            target_layer,
            device,
            cam_smoother,
            frame_idx,
            THRESHOLD,
            label_smoother,
            label_stabilizer
            )

        # Initialize video writer once output frame size is known
        if writer is None:
            h, w = overlayed.shape[:2]
            writer = create_video_writer(
                output_path=output_path,
                fps=fps,
                frame_size=(w, h)
            )

        writer.write(overlayed)

    # Release video resources
    cap.release()
    if writer is not None:
        writer.release()

    print("[INFO] Finished video explanation")
    print("[INFO] Saved to:", output_path)



if __name__ == "__main__":
    
    # Instantiate a Argument-Parser object
    # Description is displayed for -h or --help in the terminal
    parser = argparse.ArgumentParser(description="Grad-CAM XAI for video")
    parser.add_argument(
        "video_path",
        type=str,
        nargs="?",
        default=str(VIDEO_PATH),
        help="Path to input video file"
    )

    # Read the argument from the terminal
    args = parser.parse_args()

    # Pass the video path to the main function
    main(args.video_path)