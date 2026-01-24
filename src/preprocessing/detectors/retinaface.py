from insightface.app import FaceAnalysis
import warnings


class RetinaFaceDetector:
    
    def __init__(self, device = "cuda", info = True):
        
        ctx_id = 0 if device == "cuda" else -1

        if device == "cuda":
            providers = ["CUDAExecutionProvider"] 
        else: 
            providers = ["CPUExecutionProvider"]

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message = "Specified provider.*")
    
            self.app = FaceAnalysis(
                allowed_modules = ["detection"],
                providers = providers
            )

            self.app.prepare(ctx_id = ctx_id, det_size = (640, 640))

        if info:
            print(f"[INFO] Detector initialized on: {providers[0]}")

    def detect_face(self, image):
        
        faces = self.app.get(image)

        detected_faces = []

        for f in faces:

            x1, y1, x2, y2 = map(int, f.bbox)
            
            detected_faces.append({
                "box": (x1, y1, x2-x1, y2-y1),
                "confidence": float(f.det_score),
                "eyes": {
                    "left_eye": tuple(map(int, f.kps[0])),
                    "right_eye": tuple(map(int, f.kps[1]))
                }
            })

        return detected_faces
