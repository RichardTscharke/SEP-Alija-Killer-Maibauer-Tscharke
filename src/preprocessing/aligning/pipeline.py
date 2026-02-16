from copy import deepcopy
from .clip import clip_face
from .crop import crop_face
from .align import align_face
from ..debug.visualize2 import visualize


def preprocess_image(sample, vis = False):
    """
    Alignment pipeline which assumes that face detection and validation of face and sample already happened.
    -> two eye requirement, box format: (x, y, w, h), image forma: BGR. All assured by detect.py
    The vis flags is meant as a debugging tool or visualization of our pipeline.
    """
    
    # Stores the stages as images for a visualization of our alignment pipeline
    stages = [] if vis else None

    if sample is None:
        return None

    if vis:
        stages.append(("original", deepcopy(sample)))

    # Enlarges the bounding box by the desired clip ratio to reduce border artifacts
    sample = clip_face(sample, clip_ratio=0.4)
    if sample is None:
        return None
    if vis:
        stages.append(("clipped", deepcopy(sample)))

    # Crops the image around the clipped bounding box
    sample = crop_face(sample)
    if sample is None:
        return None
    if vis:
        stages.append(("cropped", deepcopy(sample)))

    # Aligns the image via an affine transformation which rotates and scales
    # the eyes of each image to fixed reference points
    sample = align_face(sample)
    if sample is None:
        return None
    if vis:
        stages.append(("aligned", deepcopy(sample)))

    # Plots a graphic of all stages via matplotlib
    if vis:
        visualize(stages)

    return sample
