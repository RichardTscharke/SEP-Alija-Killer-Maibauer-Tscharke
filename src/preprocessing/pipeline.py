import numpy as np
from copy import deepcopy
from .clip import clip_face
from .crop import crop_face
from .align import align_face
from .debug.visualize import visualize


def preprocess_image(image,
                     detector,
                     do_cliping = True,
                     do_cropping = True,
                     do_aligning = True,
                     vis = False,
                     debug = False
                     ):

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

    if w <= 0 or h <= 0:
        return None

    keypoints = {}

    for k, (kx, ky) in face["keypoints"].items():

        if not np.isfinite(kx) or not np.isfinite(ky):
            return None
        
        if (kx < x or kx > x + w) or (ky < y or ky > y + h):
            return None
        
        keypoints[k] = int(kx), int(ky)

    sample = {

        "image": np.array(image),
        "box": (x, y, w, h),
        "keypoints": keypoints
    }

    if vis:
        stages.append(("original", deepcopy(sample)))


    if do_cliping:
        sample = clip_face(sample)

        if not is_valid_sample(sample):
            return None
        
        if vis:
            stages.append(("clipped", deepcopy(sample)))


    if do_cropping:
        sample = crop_face(sample)

        if not is_valid_sample(sample):
            return None
        
        if vis:
            stages.append(("cropped", deepcopy(sample)))


    if do_aligning:
        sample = align_face(sample)

        if not is_valid_sample(sample):
            return None
        
        if vis:
            stages.append(("aligned", deepcopy(sample)))


    if vis:
        visualize(stages)        

    return sample


def is_valid_sample(sample):
    return (
        isinstance(sample, dict)
        and "image" in sample
        and "box" in sample
        and "keypoints" in sample
        and sample["image"] is not None
        )