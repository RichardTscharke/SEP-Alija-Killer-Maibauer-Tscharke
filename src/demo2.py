import torch
from demo.cam import Webcam
from demo.draw_box_landmarks import set_colors
from demo.face_tracker import FaceTracker
from demo.face_process_pipeline import FaceStreamProcessor
from preprocessing.detectors.retinaface import RetinaFaceDetector

from models.ResNetLight2 import ResNetLightCNN2

BOX_COLOR       = (0, 0, 255) # BGR
KEYPOINTS_COLOR = (0, 255, 0) # BGR

VERSION = "models/ResNetLight_v2.pth" # Make sure this is the latest model path

def load_model(weight_path, device):
    model = ResNetLightCNN2(num_classes = 6)
    state = torch.load(weight_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def main():

    set_colors(BOX_COLOR, KEYPOINTS_COLOR)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(VERSION, device)

    detector = RetinaFaceDetector(device = "cpu") # Detector is used to update the box+landmarks to groundtruth every n-th frame

    tracker = FaceTracker() # Tracker is used to predict the box movement between every n-th frame

    demo = FaceStreamProcessor(model, detector, tracker, detect_every_n = 5) # Detect face every n frames. FaceStreamProcessor holds the logic for how each frame is processed.

    Webcam.run(demo.process_frame) # Webcam is a simple webcam loop. For every frame it calls demo.process_frame 

if __name__ == "__main__":
    main()
