import numpy as np
from .validate import is_valid_face
from .pipeline import preprocess_image

def detect_and_preprocess(image, detector, detect_only = False, **kwargs):
    """
    Detects faces, selects the highest-confidence valid face and optionally applies alignment / preprocessing.
    Returns None if no valid face is found.
    """
    faces = detector.detect_face(image)
    if faces is None or len(faces) == 0:
        return None

    # Choose one face in the image based on the highest confidence
    faces = sorted(faces, key=lambda f: f["confidence"], reverse=True)
    
    # Check if the bounding box and both eyes are within the image
    face = next((f for f in faces if is_valid_face(f)), None)

    if face is None:
        return None

    sample = build_sample_from_face(image, face)

    # Demo case
    if detect_only:
        return sample
    
    # Pipeline case: Validate -> Clip -> Crop -> Align
    return preprocess_image(sample, **kwargs)

def build_sample_from_face(image, face):
    """
    Builds a standardized sample dictionary expected by the preprocessing pipeline.
    Assumes bounding boxes are in (x, y, w, h) format and image is BGR.
    The meta field is initialized in order to store all infos about applied transformations.
    """
    return {
        "image": np.array(image),
        "box":   face["box"],
        "eyes":  face["eyes"],
        "original": face.get(
            "original",
            {
                "box": face["box"],
                "landmarks": {}
            }
        ),
        "meta": {}
    }
