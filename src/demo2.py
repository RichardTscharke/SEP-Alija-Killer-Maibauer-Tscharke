import torch
from preprocessing.detectors.retinaface import RetinaFaceDetector

from explaining.explain_utils import get_device, resolve_model_and_layer
from explaining.visualize.visualize_video.cam_smoother import CamSmoother
from explaining.visualize.visualize_video.label_smoother import LabelSmoother
from explaining.visualize.visualize_video.label_stabilizer import LabelStabilizer

from demo.cam import Webcam
from demo.face_tracker import FaceTracker
from demo.face_process_pipeline import FaceStreamProcessor

MODEL_PATH = "models/ResNetLight2_v0.pth" # Make sure this is the latest model path

TARGET_LAYER = "stage3"

DETECT_EVERY_N = 1

THRESHOLD = 0.4

def main():

    # Initialize device (GPU/CPU)
    device = get_device()

    # Load model and resolve target convolutional layer for Grad-CAM
    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, device)
    print(f"[INFO] Model loaded: {MODEL_PATH}")
    print(f"[INFO] Target Layer: {TARGET_LAYER}")

    # Initialize face detector
    detector = RetinaFaceDetector(device=device)

    # Temporal smoothing for CAMs to reduce heatmap flickering
    cam_smoother = CamSmoother(alpha=0.2, every_nth_frame=1)
    print(f"[INFO] Cam Smoother initialized for stable overlays.")

    # Temporal smoothing and stabilization for emotion label overlay
    label_smoother = LabelSmoother(alpha=0.3, every_nth_frame=5)

    # min_conf refers to the minimum confidence a prediction must achieve to be written out
    # In our case we show the top 2 classes who achieved this required confidence
    label_stabilizer = LabelStabilizer(min_conf=0.3)
    print(f"[INFO] Label Smoother & Stabilizer initialized for stable emotion labels.")


    #tracker = FaceTracker() # For now we dont use a tracker for tests

    processor = FaceStreamProcessor(model,
                                    target_layer,
                                    detector,
                                    DETECT_EVERY_N,
                                    THRESHOLD,
                                    cam_smoother,
                                    label_smoother,
                                    label_stabilizer)

    Webcam.run(processor.process_frame,
               processor.set_detect_every_n,
               processor.toggle_xai,
               processor.toggle_landmarks) 
    

if __name__ == "__main__":
    main()
