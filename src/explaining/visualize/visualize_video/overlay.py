import cv2
import numpy as np

LABELS = ["Anger", "Disgust", "Fear", "Happiness", "Sadness", "Surprise"]

def overlay_gradcam(
        image,
        cam,
        threshold=0.3,
        alpha=0.4,
        colormap=cv2.COLORMAP_JET
):
    '''
    Overlays a Grad-CAM heatmap onto an image.
    Parameters:
    - Original image (BGR, uint8, (H, W, 3))
    - Grad-CAM heatmap produced in GradCam.py class
    - threshold: minimum CAM value to be visualized
    - alpha: heatmap opacity
    - colormap: OpenCV colormap

    Returns the overlayed image in BGR and uint8.
    '''

    # Copy image for safety reasons
    img = image.copy()

    # Ensure CAM is float32
    cam = cam.astype(np.float32)

    # Apply threshold
    cam = np.clip(cam, 0, 1)
    cam[cam < threshold] = 0.0

    # Normalize
    if cam.max() > 0:
        cam /= cam.max()

    # Convert CAM to 8-bit
    cam_uint8 = np.uint8(255*cam)

    # Apply colormap
    heatmap = cv2.applyColorMap(cam_uint8, colormap)

    #Overlay heatmap onto image
    overlay = cv2.addWeighted(img, 1.0, heatmap, alpha, 0)

    return overlay

def insert_emotion_label(
        image,
        idx,
        confidence,
):  
    if idx is None:
        text = "No confident prediction"
        color = (0, 0, 255)
    else:
        text = f"{LABELS[idx]}: {confidence:.0%}"
        color = (0, 255, 0)

    h, w = image.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 900
    thickness = max(1, int(h / 450))
    #padding = int(h / 80)

    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    margin_x = int(0.04 * w)       
    margin_y = int(0.08 * h)        
    x = margin_x
    y = margin_y + text_h

    padding = int(0.4 * text_h)

    border = 5
    cv2.rectangle(
        image,
        (x - padding - border, y - text_h - padding - border),
        (x + text_w + padding + border, y + padding + border),
        (230, 230, 230),
        -1,
    )

    cv2.rectangle(
        image,
        (x - padding, y - text_h - padding),
        (x + text_w + padding, y + padding),
        (79, 79, 47),
        -1,
    )

    cv2.putText(
        image,
        text,
        (x, y),
        font,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA,
    )

    return image