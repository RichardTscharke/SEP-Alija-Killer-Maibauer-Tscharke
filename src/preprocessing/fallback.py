import cv2
import numpy as np
from copy import deepcopy
from .debug.visualize import visualize

def fallback(image, out_size = 64, debug = False):

    stages = []

    if debug:
        stages.append(("original", deepcopy(image)))

    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    horizontal_edges = cv2.Sobel(gray_image, cv2.CV_32F, 0, 1, ksize=3)
    visual_info = np.mean(np.abs(horizontal_edges), axis=1)

    h = image.shape[0]
    window = int(0.45 * h)    #regler

    start = np.argmax([visual_info[i : i + window].mean()
        for i in range(0, h - window)])
    
    crop = image[start : start + window, :, :]

    crop_mtcnn = cv2.resize(crop, (224, 224))

    if debug:
        stages.append(("region of info", crop.copy()))       

    h_crop, w_crop = crop.shape[:2]
    side = min(h_crop, w_crop)

    x = (w_crop - side) // 2
    y = (h_crop - side) // 2

    crop_sq = crop[y : y + side, x : x + side]

    crop64 = cv2.resize(crop_sq, (out_size, out_size))

    if debug:
        stages.append(("64x64 crop", crop64.copy()))
        visualize(stages, show_box = False, show_landmarks = False, fallback = True)

    return crop_mtcnn, crop64