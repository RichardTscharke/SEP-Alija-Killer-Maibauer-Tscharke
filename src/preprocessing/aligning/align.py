import numpy as np
import cv2

def align_face(sample, output_size = 64, target_left = (18, 16), target_right = (44, 16)):
    '''
    Affine face alignment using eye landmarks.
    The fixed target points are based on test results.
    Stores a single full inverse matrix for Grad-CAM backprojection.
    Returns aligned face of size (output_size, output_size) or None on failure.
    '''
    image = sample["image"]
    eyes = sample["eyes"]

    xL, yL = eyes["left_eye"]
    xR, yR = eyes["right_eye"]

    # Compute rotation angle
    dX = xR - xL
    dY = yR - yL

    angle = np.degrees(np.arctan2(dY, dX))

    # Calculates the euclidean length of a vector
    current_distance = np.hypot(dX, dY)

    # Degenerate eye geometry (overlapping or invalid landmarks)
    if not np.isfinite(current_distance) or current_distance <  5:
        return None
    
    # Desired distance between eyes in output
    desired_distance = np.hypot(target_right[0] - target_left[0],
                                target_right[1] - target_left[1])
    
    scale = desired_distance / current_distance

    current_center = ((xL + xR) / 2.0, (yL + yR) / 2.0)
    target_center  = ((target_left[0] + target_right[0]) / 2.0,
                      (target_left[1] + target_right[1]) / 2.0)

    # Rotation + scale around current_center
    M = cv2.getRotationMatrix2D(current_center, angle, scale)

    # Add translation to move current_center to target_center
    M[0,2] += target_center[0] - current_center[0]
    M[1,2] += target_center[1] - current_center[1]

    # Apply affine transformation (fill with black)
    sample["image"] = cv2.warpAffine(image, M,
                                     (output_size, output_size),
                                     flags=cv2.INTER_LINEAR,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0
                                     )

    # Transform the eyes to the new target points
    aligned_eyes = {}

    for key, (x, y) in eyes.items():
        aligned_eyes[key] = transform_point((x, y), M)

    sample["eyes"] = aligned_eyes    
    sample["box"] = (0, 0, output_size, output_size)

    # Compute full inverse for Grad-CAM: aligned -> original image
    crop_x, crop_y = sample["meta"].get("crop_offset", (0,0))
    T_crop = np.array([[1,0,-crop_x],[0,1,-crop_y],[0,0,1]], dtype=np.float32)  # 3x3
    M_aug = np.vstack([M, [0,0,1]])  # 3x3
    M_full = M_aug @ T_crop          # Original -> Aligned

    M_inv_full = cv2.invertAffineTransform(M_full[:2,:])
    sample["meta"]["affine_M"] = M_full[:2,:]
    sample["meta"]["affine_M_inv"] = M_inv_full
    sample["meta"]["aligned_size"] = output_size

    return sample

def transform_point(p, M):
    '''
    Applies a 2D affine transformation matrix to a point.
    '''
    x, y = p

    x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]

    return int(round(x_new)), int(round(y_new))