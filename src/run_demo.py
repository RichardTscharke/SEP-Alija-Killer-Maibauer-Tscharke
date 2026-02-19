from src.preprocessing.detectors.retinaface import RetinaFaceDetector
from src.explaining.explain_utils import get_device, resolve_model_and_layer, run_model
from src.explaining.visualize.visualize_video.cam_smoother import CamSmoother
from src.explaining.visualize.visualize_video.label_smoother import LabelSmoother
from src.explaining.visualize.visualize_video.label_stabilizer import LabelStabilizer
from src.demo.cam import Webcam
from src.demo.fer_controller import FERStreamController

MODEL_PATH = "model_paths/ResNetLight2_loss.pth" # Make sure this is the latest model path

TARGET_LAYER = "stage3"

DETECT_EVERY_N = 1

# Refers to the minimum confidence a prediction must achieve to be displayed in the label
MIN_CONF = 0.3

# Refers to the minimum signal strength regions must achieve in order to be displayed in the heatmap
THRESHOLD = 0.4

def main():
    '''
    Entry point for the live FER demo.
    Pipeline:
    - Load device, model and target layer
    - Initialize retina face detector and smoothing utilities
    - Process video frame-by-frame with landmarks and/or Grad-CAM overlays as well as labels
    - Write annotated frames to output video

    Keys:
    'q'    : Quit demo
    'h'    : Heatmap overlay by Grad-CAM
    'k'    : Keypoints overlay by RetinaFaceDetector
    '1'-'9': Detect every n-th frame 
    '''

    # Initialize device (GPU/CPU)
    device = get_device()
    print(f"[INFO] Grad-CAM inference will be calculated on device: {device}")

    # Load model and resolve target convolutional layer for Grad-CAM
    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, device)
    print(f"[INFO] Model: {MODEL_PATH} | Target Layer: {TARGET_LAYER}")

    # Initialize face detector
    detector = RetinaFaceDetector(device=device)

    # Temporal smoothing for CAMs to reduce heatmap flickering
    cam_smoother = CamSmoother(alpha=0.2, every_nth_frame=1)
    print(f"[INFO] Cam Smoother initialized for stable overlays.")

    # Temporal smoothing and stabilization for emotion label overlay
    label_smoother = LabelSmoother(alpha=0.3, every_nth_frame=5)

    # min_conf refers to the minimum confidence a prediction must achieve to be written out
    # In our case we show the top 2 classes who achieved this required confidence
    label_stabilizer = LabelStabilizer(min_conf=MIN_CONF)
    print(f"[INFO] Label Smoother & Stabilizer initialized for stable emotion labels.")

    # FERStreamController orchestrates:
    # detection -> alignment -> inference -> Grad-CAM -> smoothing -> rendering
    processor = FERStreamController(device,
                                    model,
                                    target_layer,
                                    detector,
                                    DETECT_EVERY_N,
                                    THRESHOLD,
                                    cam_smoother,
                                    label_smoother,
                                    label_stabilizer,
                                    run_model)

    # Webcam orchestrates:
    # frame processing, dynamic detection frequency adjustment, toggling Grad-CAM, toggling bounding box + landmarks
    Webcam.run(processor.process_frame,
               processor.set_detect_every_n,
               processor.toggle_xai,
               processor.toggle_landmarks) 
    

if __name__ == "__main__":
    main()
