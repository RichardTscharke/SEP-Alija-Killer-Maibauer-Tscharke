import cv2
import torch
import numpy as np
import torch.nn.functional as F
from .explain_gradcam import explain_gradcam
from preprocessing.aligning.detect import detect_and_preprocess

emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

def load_model(model_class, weight_path, device, num_classes = len(emotions)):

    model = model_class(num_classes=num_classes)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    #model.eval()
    return model


def np_to_tensor(np_img, device):
    np_img = np_img[:, :, ::-1].astype("float32") / 255.0
    tensor = torch.from_numpy(np_img).permute(2, 0, 1)
    tensor = (tensor - 0.5) / 0.5
    return tensor.unsqueeze(0).to(device)

def preprocess_image(image_path, detector, device):
    '''
    Full preprocessing for a single image.
    Returns: sample dict as defined in preprocessing/aligning/detect.py
    '''
    original_img = cv2.imread(str(image_path))

    sample = detect_and_preprocess(original_img, detector)

    sample["original_img"] = original_img
    sample["input_tensor"] = np_to_tensor(sample["image"], device)

    return sample

def cam_to_original(cam_aligned, meta, original_shape):
    '''
    Projects CAM from aligned-face space back to original image space.
    '''
    h_orig, w_orig = original_shape[:2]

    aligned_size = meta["aligned_size"]
    
    cam_resized = cv2.resize(
        cam_aligned.astype(np.float32),
        (aligned_size, aligned_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Apply the full inverse affine (including crop offset) directly to original image size
    M_inv = meta["affine_M_inv"]
    cam_orig_crop = cv2.warpAffine(
        cam_resized,
        M_inv,
        (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    return cam_orig_crop


def run_inference(sample, model, target_layer):
    cam_aligned, logits = explain_gradcam(
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

    print(
    "cam aligned:", cam_aligned.min(), cam_aligned.max(),
    "cam original:", cam_original.min(), cam_original.max(),
    "nonzero:", np.count_nonzero(cam_original)
    )

    # Update sample directory
    sample["cam"] = cam_aligned
    sample["cam_original"] = cam_original
    sample["logits"] = logits
    sample["probs"] = probs

    return sample
