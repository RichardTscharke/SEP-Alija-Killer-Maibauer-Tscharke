import numpy as np

def is_valid_face(face, debug = False):

    x, y, w, h = face["box"]

    if w <= 0 or h <= 0:
        return False
    
    for k, (kx, ky) in face["eyes"].items():

        if not np.isfinite(kx) or not np.isfinite(ky):
            if debug:
                print(f"eye missing: {k}")

            return False
        
        if (kx < x or kx > x + w) or (ky < y or ky > y + h):
            if debug:
                print(f"eye out of box: {k}")

            return False
    
    return True
        

def is_valid_sample(sample):
    return (
        isinstance(sample, dict)
        and "image" in sample
        and "box" in sample
        and "eyes" in sample
        and sample["image"] is not None
        )