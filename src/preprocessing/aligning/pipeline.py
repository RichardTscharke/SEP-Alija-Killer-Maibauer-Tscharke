import numpy as np
from copy import deepcopy
from .validate import is_valid_face, is_valid_sample
from .clip import clip_face
from .crop import crop_face
from .align import align_face
from ..debug.visualize import visualize


def preprocess_image(image,
                     detector,
                     do_clipping = True,
                     do_cropping = True,
                     do_aligning = True,
                     vis = False,
                     debug = False
                     ):

    stages = []

    try:
        faces = detector.detect_face(image)
    except Exception as e:
        if debug:
            print(f"[preprocess_image] detect_faces failed: {e}")
        return None

    if faces is None or len(faces) == 0:
        return None

    faces = sorted(faces, key=lambda f: float(f["confidence"]), reverse=True)

    face = next((f for f in faces if is_valid_face(f)), None)

    if face is None:
        return None

    sample = {

        "image": np.array(image),
        "box": face["box"],
        "eyes": face["eyes"]
    }

    if vis:
        stages.append(("original", deepcopy(sample)))


    if do_clipping:
        sample = clip_face(sample, clip_ratio = 0.4)

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