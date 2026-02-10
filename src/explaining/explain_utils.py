import cv2
import torch
from preprocessing.aligning.detect import detect_and_preprocess


def load_model(model_class, weight_path, device, num_classes):

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    return model

def preprocess_frame(frame, detector):
    '''
    Runs face detection and alignment on a single frame.
    Returns a sample dict containing:
    - Aligned face image
    - Original image
    - Meta informaton for backprojection
    '''
    sample = detect_and_preprocess(frame, detector)

    sample["original_img"] = frame

    return sample

def preprocess_image(image_path, detector):
    '''
    Wrapper function for an image as input instead of a frame.
    '''
    frame = cv2.imread(str(image_path))
    return preprocess_frame(frame, detector)


