from demo.cam import Webcam
from demo.draw_box_landmarks import set_colors
from demo.face_tracker import FaceTracker
from demo.face_process_pipeline import FaceStreamProcessor
from preprocessing.detectors.retinaface import RetinaFaceDetector

BOX_COLOR       = (0, 0, 255) # BGR
KEYPOINTS_COLOR = (0, 255, 0) # BGR

def main():

    set_colors(BOX_COLOR, KEYPOINTS_COLOR)

    detector = RetinaFaceDetector(device = "cpu") # Detector is used to update the box+landmarks to groundtruth every n-th frame

    tracker = FaceTracker() # Tracker is used to predict the box movement between every n-th frame

    demo = FaceStreamProcessor(detector, tracker, detect_every_n = 5) # Detect face every n frames. FaceStreamProcessor holds the logic for how each frame is processed.

    Webcam.run(demo.process_frame) # Webcam is a simple webcam loop. For every frame it calls demo.process_frame 

if __name__ == "__main__":
    main()
