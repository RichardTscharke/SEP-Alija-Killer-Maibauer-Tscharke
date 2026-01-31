from demo.cam import Webcam
from demo.face_process_pipeline import FaceStreamProcessor

def main():
    demo = FaceStreamProcessor(detect_every_n = 5) # Detect face every n frames. FaceStreamProcessor holds the logic for how each frame is processed.

    Webcam.run(demo.process_frame) # Webcam is a simple webcam loop. For every frame it calls demo.process_frame 

if __name__ == "__main__":
    main()
