import os
import sys
from insightface.app import FaceAnalysis
from contextlib import contextmanager
import warnings


class RetinaFaceDetector:
    
    def __init__(self, device = "cuda", info = True):
        
        ctx_id = 0 if device == "cuda" else -1

        if device == "cuda":
            providers = ["CUDAExecutionProvider"] 
        else: 
            providers = ["CPUExecutionProvider"]

        with suppress_stdout():               # surpresses prints everytime a detector is initialized
            self.app = FaceAnalysis(
                allowed_modules=["detection"],
                providers=providers
            )
            self.app.prepare(ctx_id = ctx_id, det_size = (640, 640))

        if info:
            print(f"[INFO] Detector initialized on: {providers[0]}")

    def detect_face(self, image):
        
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
                "original": {
                    "box": (x1, y1, x2 - x1, y2 - y1),
                    "landmarks": landmarks,
                },
            })

        return detected_faces
    
@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
