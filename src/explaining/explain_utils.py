import cv2
import torch
import numpy as np
from .cam.cam_runner import run_gradcam
from .cam.cam_projection import cam_to_original
from preprocessing.aligning.detect import detect_and_preprocess

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

def load_model(model_class, weight_path, device, num_classes = len(emotions)):

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    #model.eval()
    return model

def np_to_tensor(np_img, device):
    '''
    Converts an OpenCV BGR image to a normalized torch tensor:
    shape: (1, 3, H, W), range approx [-1, 1]
    '''
    np_img = np_img[:, :, ::-1].astype("float32") / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)

def preprocess_frame(frame, detector, device):
    '''
    Runs face detection, alignment and tensor conversion on a single frame.
    Returns a sample dict as defined by the preprocessing pipeline.
    '''
    sample = detect_and_preprocess(frame, detector)

    sample["original_img"] = frame
    sample["input_tensor"] = np_to_tensor(sample["image"], device)

    return sample

def preproess_img(image_path, detector, device):
    frame = cv2.imread(str(image_path))
    return preprocess_frame(frame, detector, device)

def run_explanation(sample, model, target_layer):
    cam_aligned, logits = run_gradcam(
        model=model,
        input_tensor=sample["input_tensor"],
        target_layer=target_layer
    )

    probs = torch.softmax(logits, dim=1)[0].cpu().detach().numpy()

    # Paste CAM back into the original image
    cam_original = cam_to_original(
        cam_aligned=cam_aligned,
        meta=sample["meta"],
        original_shape=sample["original_img"].shape
    )

    # Update sample directory
    sample["cam"] = cam_aligned
    sample["cam_original"] = cam_original
    sample["logits"] = logits
    sample["probs"] = probs

    return sample
