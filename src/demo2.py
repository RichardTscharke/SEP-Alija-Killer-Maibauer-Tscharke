import torch
from demo.cam import Webcam
from demo.draw_box_landmarks import set_colors
from demo.face_tracker import FaceTracker
from demo.face_process_pipeline import FaceStreamProcessor
from preprocessing.detectors.retinaface import RetinaFaceDetector

from explaining.explain_utils import load_model

from models.ResNetLight2 import ResNetLightCNN2

BOX_COLOR       = (0, 0, 255) # BGR
KEYPOINTS_COLOR = (0, 255, 0) # BGR

MODEL_PATH = "models/ResNetLight2_v0.pth" # Make sure this is the latest model path


def main():

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    set_colors(BOX_COLOR, KEYPOINTS_COLOR)

    model  = load_model(
        model_class=ResNetLightCNN2,
        model_path=MODEL_PATH,
        device=device,
        num_classes=6
    )

    detector = RetinaFaceDetector(device=device) # Detector is used to update the box+landmarks to groundtruth every n-th frame

    tracker = FaceTracker() # Tracker is used to predict the box movement between every n-th frame

    demo = FaceStreamProcessor(model, detector, detect_every_n=5) # Detect face every n frames. FaceStreamProcessor holds the logic for how each frame is processed.

    Webcam.run(demo.process_frame) # Webcam is a simple webcam loop. For every frame it calls demo.process_frame 

if __name__ == "__main__":
    main()
