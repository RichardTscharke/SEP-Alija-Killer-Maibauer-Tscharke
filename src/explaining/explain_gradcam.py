import cv2
import torch
import torch.nn.functional as F

from explaining.GradCam import GradCAM
from explaining.explain_utils import np_to_tensor
from preprocessing.detectors import RetinaFaceDetector
from preprocessing.aligning.detect import detect_and_preprocess


def explain_gradcam(
    model,
    image_path,
    target_layer,
    device,
    ):
    '''
    Computes Grad-CAM for a single image.
    Pipeline:
    - face detection & alignment
    - forward pass
    - Grad-CAM generation for predicted class

    Returns:
    - original image
    - aligned face image
    - CAM heatmap
    - class probabilities
    '''

    # Evaluate model
    model.eval()

    # Load image
    original_img = cv2.imread(str(image_path))
    if original_img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # XAI assumes face detection and alignment via RetinaFace
    detector = RetinaFaceDetector(device=str(device))

    # Returns aligned face + metadata (bounding box, etc.)
    sample = detect_and_preprocess(original_img, detector)
    aligned_img = sample["image"]

    # Convert aligned image to input tensor
    input_tensor = np_to_tensor(aligned_img, device)

    # Forward pass & class probabilities
    grad_cam = GradCAM(target_layer)

    logits = model(input_tensor)

    # No gradients since only probabilities are needed for the  visualization
    with torch.no_grad():
        probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = probs.argmax()

    # Generate Grad-CAM for predicted class
    cam = grad_cam.generate(logits, pred_class)

    # Remove used forward and backward hooks
    grad_cam.remove()

    return {
        "original_img": original_img,
        "aligned_img": aligned_img,
        "cam": cam,
        "probs": probs,
        "pred_class": pred_class,
        "box": sample["original"]["box"]
    }
