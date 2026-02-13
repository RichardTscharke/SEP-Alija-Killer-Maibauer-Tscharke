import cv2
import numpy as np

def cam_to_original(cam, meta, original_shape):
    '''
    Projects a Grad-CAM heatmap from aligned-face coordinates back into original image coordinates.
    Steps:
    - Upscale CAM from layer resolution (e.g. 4x4 in stage3) to aligned-face resolution (64x64).
    - Apply the full inverse affine transformation computated and stored by alignment pipeline.
    - Return CAM in original image resolution.
    '''

    # Target resolution of the aligned face (for us 64x64)
    aligned_size = meta["aligned_size"]
    
    # Resize CAM from feature-map resolution to aligned-face resolution
    cam_aligned_space = cv2.resize(
        cam.astype(np.float32),
        (aligned_size, aligned_size),
        interpolation=cv2.INTER_LINEAR
    )

    # Inverse affine transform mapping (aligned face -> original image) 
    M_inv = meta["affine_M_inv"]

    # Original frame height and width
    h_orig, w_orig = original_shape[:2]

    # Warp resized CAM into original image coordinates
    cam_orig_crop = cv2.warpAffine(
        cam_aligned_space,
        M_inv,
        (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # CAM in original image resolution and coordinate system
    return cam_orig_crop, cam