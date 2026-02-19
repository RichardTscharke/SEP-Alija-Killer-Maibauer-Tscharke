import copy
from pathlib import Path
from src.preprocessing.detectors.retinaface import RetinaFaceDetector

from src.explaining.explain_utils import get_device, resolve_model_and_layer, preprocess_image
from src.explaining.cam.explain_frame import explain_frame

IMAGE_PATH = Path("/Users/richardachtnull/Desktop/IMG_0477.jpg")

MODEL_PATH1   = "model_paths/RafCustom_v0.pth"
MODEL_PATH2   = "model_paths/ResNetLight2_v8.pth"

TARGET_LAYER1 = "features.3"
TARGET_LAYER2 = "stage3.conv2"

THRESHOLD    = 0.4

SAVE_PATH = "/Users/richardachtnull/Desktop/figure.pdf"

import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_overlay(original_img, cam_original, threshold=0.4):
    """
    Creates only the final Grad-CAM overlay in original image space.
    Returns RGB numpy image.
    """

    cam_original = np.clip(cam_original, 0, 1)

    heat = np.uint8(255 * cam_original)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

    original_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

    mask = cam_original >= threshold
    overlay = original_rgb.copy()
    overlay[mask] = (
        0.6 * overlay[mask] + 0.4 * heat[mask]
    ).astype(np.uint8)

    return overlay


def visualize_two_models(label1,
                         label2,
                         conf1,
                         conf2,
                         original_img,
                         cam_model1,
                         cam_model2,
                         threshold=0.4,
                         save_path=None,
                         ):
    """
    Creates a clean 2-subplot figure for LaTeX reports.
    """

    overlay1 = create_overlay(original_img, cam_model1, threshold)
    overlay2 = create_overlay(original_img, cam_model2, threshold)

    plt.rcParams["font.family"] = "Times New Roman"

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(overlay1)
    axes[0].axis("off")
    axes[0].text(
        0.02, 0.95,
        f"Pred: {label1} ({conf1:.1f}%)",
        transform=axes[0].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    axes[1].imshow(overlay2)
    axes[1].axis("off")
    axes[1].text(
        0.02, 0.95,
        f"Pred: {label2} ({conf2:.1f}%)",
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )


    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    else:
        plt.show()

    plt.close()



def main():
   
    image_path = Path(IMAGE_PATH)

    device = get_device()
    
    model1, target_layer1 = resolve_model_and_layer(MODEL_PATH1, TARGET_LAYER1, device, "raf")
    model2, target_layer2 = resolve_model_and_layer(MODEL_PATH2, TARGET_LAYER2, device, "rnl")

    detector = RetinaFaceDetector(device=device)


    sample = preprocess_image(image_path, detector, device)

    sample1 = explain_frame(
        sample=copy.deepcopy(sample),
        model=model1,
        target_layer=target_layer1
    )

    sample2 = explain_frame(
        sample=copy.deepcopy(sample),
        model=model2,
        target_layer=target_layer2
    )

    pred1 = np.argmax(sample1["probs"])
    pred2 = np.argmax(sample2["probs"])

    emotions = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

    label1 = emotions[pred1]
    label2 = emotions[pred2]

    conf1 = sample1["probs"][pred1] * 100
    conf2 = sample2["probs"][pred2] * 100


    visualize_two_models(
    label1,
    label2,
    conf1,
    conf2,
    original_img=sample["original_img"],
    cam_model1=sample1["cam_original"],
    cam_model2=sample2["cam_original"],
    threshold=THRESHOLD,
    save_path=SAVE_PATH,
    )
    


if __name__ == "__main__":
    
    main()
