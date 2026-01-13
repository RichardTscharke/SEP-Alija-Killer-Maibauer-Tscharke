import numpy as np
from copy import deepcopy
from .clip import clip_face
from .crop import crop_face
from .align import align_face
from .debug.visualize import visualize


def preprocess_image(image, detector, do_cliping = True, do_cropping = True, do_aligning = True, debug = False):

    stages = []

    try:
        faces = detector.detect_faces(image)
    except Exception as e:
        if debug:
            print(f"[preprocess_image] detect_faces failed: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None


    face = max(faces, key=lambda f: float(f["confidence"]))

    x, y, w, h = face["box"]

    keypoints = {}

    for key, (x1, y1) in face["keypoints"].items():

        keypoints[key] = int(x1), int(y1)

    sample = {

        "image": np.array(image),
        "box": (x, y, w, h),
        "keypoints": keypoints
    }

    if debug:
        stages.append(("original", deepcopy(sample)))

    if do_cliping:
        sample = clip_face(sample)
        if debug:
            stages.append(("padded", deepcopy(sample)))

    if do_cropping:
        sample = crop_face(sample)
        if debug:
            stages.append(("cropped", deepcopy(sample)))

    if do_aligning:
        sample = align_face(sample)
        if debug:
            stages.append(("aligned", deepcopy(sample)))

    if debug:
        visualize(stages)        

    return sample
