import os
import sys
from insightface.app import FaceAnalysis
from contextlib import contextmanager


class RetinaFaceDetector:
    """
    Thin wrapper around InsightFace RetinaFace detector.
    Provides face detection with landmarks and bounding boxes
    normalized to (x, y, w, h) format.
    """
    def __init__(self, device = "cuda", info = True):

        if device == "cuda":
            ctx_id = 0
            providers = ["CUDAExecutionProvider"] 
        else: 
            ctx_id = -1
            providers = ["CPUExecutionProvider"]

        # Silence InsightFace model-loading output during initialization
        with suppress_stdout(): 
            self.app = FaceAnalysis(
                allowed_modules=["detection"],
                providers=providers
            )
            self.app.prepare(ctx_id = ctx_id, det_size = (640, 640))

        if info:
            print(f"[INFO] Detector initialized on: {providers[0]}")

    def detect_face(self, image):
        """
        Expects a valid image array compatible with insightface (BGR, HWC).
        Returns a list of detected faces. Each face contains:
        - box: (x, y, w, h) in image pixel coordinates
        - confidence: detection score
        - eyes: left/right eye landmarks
        - original: raw detector output for downstream usage (e.g. demo)
        """
        faces = self.app.get(image)

        detected_faces = []

        for f in faces:

            landmarks = {"left_eye":    tuple(map(int, f.kps[0])),
                         "right_eye":   tuple(map(int, f.kps[1])),
                         "nose":        tuple(map(int, f.kps[2])),
                         "left_mouth":  tuple(map(int, f.kps[3])),
                         "right_mouth": tuple(map(int, f.kps[4])),
            }

            x1, y1, x2, y2 = map(int, f.bbox)
            
            detected_faces.append({
                "box": (x1, y1, x2-x1, y2-y1),
                "confidence": float(f.det_score),
                "eyes": {
                    "left_eye":  landmarks["left_eye"],
                    "right_eye": landmarks["right_eye"],
                },
                # Our alignment manipulates the box and landmarks
                # Therefore, store an original version of the box and landmarks for the demo. 
                "original": {
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "landmarks": landmarks,
                },
            })

        return detected_faces
    
@contextmanager
def suppress_stdout():
    """
    Temporarily suppresses all output written to sys.stdout.
    Used to silence verbose third-party libraries (e.g. InsightFace) during model initialization.
    Stdout is restored even if an exception occurs.
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
