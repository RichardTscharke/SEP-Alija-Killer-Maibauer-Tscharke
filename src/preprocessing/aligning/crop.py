def crop_face(sample):
    '''
    Crops the image to the current face bounding box and updates landmarks accordingly.
    After cropping the image coordinaten system is reset such that the top-left corner equals (0, 0).
    '''
    image = sample["image"]
    x, y, w, h = sample["box"]

    h_img, w_img = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    # Stored to invert the transformation later in regards of Grad-CAM overlay
    sample.setdefault("meta", {})
    sample["meta"]["crop_offset"] = (x1, y1)
    sample["meta"]["crop_shape"] = (y2 - y1, x2 - x1)

    # Crop equals the part of the original image bounded by the box
    sample["image"] = image[y1:y2, x1:x2]

    # Update the eyes coordinates to the crop
    new_eyes = {}
    
    for key, (kx, ky) in sample["eyes"].items():
        new_eyes[key] = (kx - x1, ky - y1)

    sample["eyes"] = new_eyes

    # Ensures that the eyes are not outside of the box after cropping
    for k, (kx, ky) in sample["eyes"].items():
        if kx < 0 or ky < 0 or kx >= (x2 - x1) or ky >= (y2 - y1):
            return None

    # The top-left coordinates of the box are now (0, 0)
    sample["box"] = (0, 0, x2 - x1, y2 - y1)

    return sample