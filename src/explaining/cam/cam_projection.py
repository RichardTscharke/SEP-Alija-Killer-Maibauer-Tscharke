import cv2
import numpy as np

def cam_to_original(cam, meta, original_shape):
    '''
    Projects a Grad-CAM heatmap from layer-space back into original image coordinates.
    Steps:
    1. Upscale CAM from layer resolution (e.g. 4x4 in stage3) to aligned-face resolution (64x64).
    2. Apply the full inverse affine transformation compuatated and stored by alignment pipeline.
    3. Return CAM in original image resolution.
    '''

    # 1. Resize CAM to match the aligned face image resolution
    aligned_size = meta["aligned_size"]
    
    cam_aligned_space = cv2.resize(
        cam.astype(np.float32),
        (aligned_size, aligned_size),
        interpolation=cv2.INTER_LINEAR
    )

    # 2. Warp CAM back into original image coordinates using inverse affine transform
    M_inv = meta["affine_M_inv"]
    h_orig, w_orig = original_shape[:2]

    cam_orig_crop = cv2.warpAffine(
        cam_aligned_space,
        M_inv,
        (w_orig, h_orig),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # 3. Return CAM in original image resolution
    return cam_orig_crop