import torch
from demo.cam import Webcam
from demo.draw_box_landmarks import set_colors
from demo.face_tracker import FaceTracker
from demo.face_process_pipeline import FaceStreamProcessor
from preprocessing.detectors.retinaface import RetinaFaceDetector

from explaining.explain_utils import resolve_model_and_layer

from models.ResNetLight2 import ResNetLightCNN2

BOX_COLOR       = (0, 0, 255) # BGR
KEYPOINTS_COLOR = (0, 255, 0) # BGR

MODEL_PATH = "models/ResNetLight2_v0.pth" # Make sure this is the latest model path

TARGET_LAYER = "stage3"


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    set_colors(BOX_COLOR, KEYPOINTS_COLOR)

    model, target_layer = resolve_model_and_layer(MODEL_PATH, TARGET_LAYER, device)

    detector = RetinaFaceDetector(device=device) # Detector is used to update the box+landmarks to groundtruth every n-th frame

    #tracker = FaceTracker() # Tracker is used to predict the box movement between every n-th frame

    processor = FaceStreamProcessor(model, detector, detect_every_n=1, target_layer=target_layer) # Detect face every n frames. FaceStreamProcessor holds the logic for how each frame is processed.

    Webcam.run(process_frame=processor.process_frame,
               set_detect_every_n=processor.set_detect_every_n,
               toggle_xai=processor.toggle_xai,
               toggle_landmarks=processor.toggle_landmarks) 

if __name__ == "__main__":
    main()
