import cv2
import numpy as np

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