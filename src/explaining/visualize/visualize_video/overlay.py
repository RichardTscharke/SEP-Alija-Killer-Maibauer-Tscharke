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

    # Copy image to avoid in-place modification
    img = image.copy()

    # Ensure CAM uses float32 for OpenCV operations
    cam = cam.astype(np.float32)

    # Apply threshold to suppress weak activations
    cam = np.clip(cam, 0, 1)
    cam[cam < threshold] = 0.0

    # Re-normalize CAM after thresholding
    if cam.max() > 0:
        cam /= cam.max()

    # Convert normalized CAM to 8-bit for color mapping
    cam_uint8 = np.uint8(255*cam)

    # Map CAM values to a color heatmap
    heatmap = cv2.applyColorMap(cam_uint8, colormap)

    # Blend heatmap with original image
    overlay = cv2.addWeighted(img, 1.0, heatmap, alpha, 0)

    return overlay

def insert_emotion_label(image, predictions):  
    '''
    Draws the predicted emotion label and confidence onto the image.
    Either two/one/no top candidate(s) exist.
    Modifies the image in-place and returns it.
    '''
    
    # Image dimensions used for resolution-independet scaling
    h, w = image.shape[:2]

    # Font parameters scaled relative to image height
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = h / 900
    thickness = max(1, int(h / 450))

    # Select label text and color based on prediction availability
    lines = []
    for i, pred in enumerate(predictions):
        if pred is None:
            continue
        idx, conf = pred
        color = (0, 255, 0) if i == 0 else (0, 165, 255)
        lines.append((f"{LABELS[idx]}: {conf:.0%}", color))

    if not lines:
        lines = (["No confident prediction"], (0, 0, 255))

    # Measure text box
    text_sizes = [
        cv2.getTextSize(txt, font, font_scale, thickness)[0]
        for txt, _ in lines
    ]
    txt_w = max(w for w, _ in text_sizes)
    txt_h = sum(h for _, h in text_sizes)

    line_spacing = int(0.4 * text_sizes[0][1])
    padding = int(0.6 * text_sizes[0][1])

    box_w = txt_w + 2 * padding
    box_h = txt_h + padding * 2 + line_spacing * (len(lines) - 1)

    # Top-left corner of label box
    x = int(0.04 * w)
    y = int(0.08 * h)

    # Draw background rectangle
    border = 5
    cv2.rectangle(
        image,
        (x, y),
        (x + box_w, y + box_h),
        (230, 230, 230),
        -1,
    )

    # Draw label rectangle
    cv2.rectangle(
        image,
        (x + 5, y + 5),
        (x + box_w - 5, y + box_h - 5),
        (79, 79, 47),
        -1,
    )

    # Draw label text
    y_txt = y + padding + text_sizes[0][1]
    for (txt, color), (_, line_h) in zip(lines, text_sizes):
        cv2.putText(
            image,
            txt,
            (x + padding, y_txt),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
        y_txt += line_h + line_spacing

    return image