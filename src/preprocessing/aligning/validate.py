import numpy as np

def is_valid_face(face, debug=False):
    """
    Validates a detected face based on bounding box sanity and eye landmark consistency.
    Requires both eyes for affine alignment.
    Expects box format (x, y, w, h).
    """
    x, y, w, h = face["box"]

    if w <= 0 or h <= 0:
        return False

    eyes = face["eyes"]

    # Precondition: Both eyes exist
    if "left_eye" not in eyes or "right_eye" not in eyes:
        if debug:
            print("missing one or both eyes")
        return False

    # Precondition: Both eyes have valid coordinates and are within the box
    for name, (kx, ky) in eyes.items():
        if not np.isfinite(kx) or not np.isfinite(ky):
            if debug:
                print(f"eye invalid: {name}")
            return False

        if not (x <= kx <= x + w and y <= ky <= y + h):
            if debug:
                print(f"eye out of box: {name}")
            return False

    return True


def is_valid_sample(sample):
    """
    Structural check for a preprocessing sample dictionary.
    Does not validate geometry or image content.
    """
    return (
        isinstance(sample, dict)
        and "image" in sample
        and "box" in sample
        and "eyes" in sample
        and sample["image"] is not None
        )