import numpy as np
from .validate import is_valid_face
from .pipeline import preprocess_image

def detect_and_preprocess(image, detector, **kwargs):

    faces = detector.detect_face(image)
    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: float(f["confidence"]), reverse=True)
    face = next((f for f in faces if is_valid_face(f)), None)

    if face is None:
        return None

    sample = build_sample_from_face(image, face)
    return preprocess_image(sample, **kwargs)

def build_sample_from_face(image, face):
    return {
        "image": np.array(image),
        "box": face["box"],
        "eyes": face["eyes"],
        "original": face.get(
            "original",
            {
                "box": face["box"],
                "landmarks": {}
            }
        )
    }
