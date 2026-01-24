import numpy as np
import cv2

def align_face(sample, output_size = 64, target_left = (18, 16), target_right = (44, 16)):

    image = sample["image"]
    eyes = sample["eyes"]

    xL, yL = eyes["left_eye"]
    xR, yR = eyes["right_eye"]

    dX = xR - xL
    dY = yR - yL

    angle = np.degrees(np.arctan2(dY, dX))

    current_distance = np.sqrt((dX ** 2) + (dY ** 2))
    desired_distance = np.linalg.norm(np.array(target_right) - np.array(target_left))

    scale = desired_distance / current_distance

    current_center = ((xL + xR) // 2, (yL + yR) // 2)
    target_center =  (target_left[0] + target_right[0]) // 2, (target_left[1] + target_right[1]) // 2

    M = cv2.getRotationMatrix2D(current_center, angle, scale)

    M[0,2] += target_center[0] - current_center[0]
    M[1,2] += target_center[1] - current_center[1]

    sample["image"] = cv2.warpAffine(image, M,
                                     (output_size, output_size),
                                     flags=cv2.INTER_AREA,
                                     borderMode=cv2.BORDER_CONSTANT,
                                     borderValue=0
                                     )

    aligned_eyes = {}

    for key, (x, y) in eyes.items():
        aligned_eyes[key] = transform_point((x, y), M)

    sample["eyes"] = aligned_eyes    
    sample["box"] = (0, 0, output_size, output_size)

    return sample

def transform_point(p, M):

    x, y = p

    x_new = M[0, 0] * x + M[0, 1] * y + M[0, 2]
    y_new = M[1, 0] * x + M[1, 1] * y + M[1, 2]

    return int(round(x_new)), int(round(y_new))