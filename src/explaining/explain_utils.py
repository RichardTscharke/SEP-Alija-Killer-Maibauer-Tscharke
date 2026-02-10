import cv2
import torch
import numpy as np
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

def cam_to_original(cam, meta, original_shape):
    '''
    Projects CAM from aligned-face space back to original image space.
    '''
    h_orig, w_orig = original_shape[:2]

    # 1. Resize CAM to aligned resolution (64x64)
    w_aligned, h_aligned = meta["aligned_size"]
    cam_aligned = cv2.resize(cam, (w_aligned, h_aligned))

    # 2. Invert affine transformation: aligned -> crop
    cam_crop = cv2.warpAffine(
        cam_aligned,
        meta["affine_M_inv"],
        (meta["crop_shape"][1], meta["crop_shape"][0]),
        flags=cv2.INTER_LINEAR
    )

    # 3. Pase into original image
    cam_orig = np.zeros((h_orig, w_orig), dtype=np.float32)
    x1, y1 = meta["crop_offset"]
    h, w = meta["crop_shape"]

    cam_orig[y1:y1+h, x1:x1+w] = cam_crop

    return cam_orig


def run_inference(sample, model, target_layer):
    cam_aligned, logits = explain_gradcam(
        model=model,
        input_tensor=sample["input_tensor"],
        target_layer=target_layer
    )

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # Paste CAM back into the original image
    cam_original = cam_to_original(
        cam=cam_aligned,
        meta=sample["meta"],
        original_shape=sample["original_img"].shape
    )

    # Update sample directory
    sample["cam"] = cam_aligned
    sample["cam_original"] = cam_original
    sample["logits"] = logits
    sample["probs"] = probs

    return sample
