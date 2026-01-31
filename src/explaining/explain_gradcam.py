import torch
import torch.nn.functional as F
import cv2

from explaining.GradCam import GradCAM
from explaining.explain_utils import np_to_tensor
from preprocessing.detectors import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess


def explain_gradcam(
    model,
    image_path,
    target_layer,
    device,
    threshold=0.4
):

    # Load image
    original_img = cv2.imread(image_path)
    if original_img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Preprocess
    detector = RetinaFaceDetector(device=str(device))
    sample = detect_and_preprocess(original_img, detector)

    aligned_img = sample["image"]
    input_tensor = np_to_tensor(aligned_img, device)

    # GradCAM
    grad_cam = GradCAM(target_layer)

    logits = model(input_tensor)
    probs = F.softmax(logits, dim=1)[0].detach().cpu().numpy()
    pred_class = probs.argmax()

    cam = grad_cam.generate(logits, pred_class)

    return {
        "original_img": original_img,
        "aligned_img": aligned_img,
        "cam": cam,
        "probs": probs,
        "pred_class": pred_class
    }
