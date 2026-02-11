import torch
from pathlib import Path
from tqdm import tqdm

from preprocessing.detectors.retinaface import RetinaFaceDetector
from explaining.explain_utils import resolve_model_and_layer
from explaining.video_utils import open_video, create_video_writer
from explaining.visualize.visualize_video.cam_smoother import CamSmoother
from explaining.visualize.visualize_video.label_smoother import LabelSmoother
from explaining.visualize.visualize_video.label_stabilizer import LabelStabilizer


from explaining.process_frame import process_frame


INPUT_PATH = Path("/Users/richardachtnull/IMG_0522.MOV")

MODEL_PATH = Path("models/ResNetLight2_v0.pth")

TARGET_LAYER = "stage3"

THRESHOLD = 0.4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(input_path):
    '''
    Runs Grad-CAM based emotion explanation on a video file.
    Takes the input path of the video and prints the path to the stored output video.
    Pipeline:
    - Load model and target layer
    - Initialize retina face detector and smoothing utilities
    - Process video frame-by-frame with Grad-CAM overlays and labels
    - Write annotated frames to output video
    '''
    
    # Define the output path
    output_path = input_path.with_name(f"{input_path.stem}_explained{input_path.suffix}")

    # Load model and resolve target convolutional layer for Grad-CAM
    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, DEVICE)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
    print(f"[INFO] Target Layer: {TARGET_LAYER}")

    # Initialize face detector
    detector = RetinaFaceDetector(device=DEVICE)
    print("[INFO] RetinaFace Detector initialized.")

    # open input video and retrieve metadata
    print(f"[INFO] Opening video: {input_path}")
    cap, fps, total_frames = open_video(input_path)
    print(f"[INFO] Video FPS (source): {fps:.2f}")
    print(f"[INFO] Total Frames: {total_frames}")

    # Temporal smoothing for CAMs to reduce heatmap flickering
    cam_smoother = CamSmoother(alpha=0.2, every_nth_frame=1)
    print(f"[INFO] Cam Smoother initialized for stable overlays.")

    # Temporal smoothing and stabilization for emotion label overlay
    label_smoother = LabelSmoother(alpha=0.3, every_nth_frame=5)

    # min_conf refers to the minimum confidence a prediction must achieve to be written out
    # In our case we show the top 2 classes who achieved this required confidence
    label_stabilizer = LabelStabilizer(min_conf=0.3)
    print(f"[INFO] Label Smoother & Stabilizer initialized for stable emotion labels.")

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
            DEVICE,
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
    main(INPUT_PATH)