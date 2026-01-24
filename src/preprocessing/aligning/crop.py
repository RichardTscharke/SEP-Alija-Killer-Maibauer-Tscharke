def crop_face(sample):

    image = sample["image"]
    x, y, w, h = sample["box"]

    h_img, w_img = image.shape[:2]

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w_img, x + w)
    y2 = min(h_img, y + h)

    sample["image"] = image[y1:y2, x1:x2]

    new_eyes = {}
    
    for key, (kx, ky) in sample["eyes"].items():
        new_eyes[key] = (kx - x1, ky - y1)

    sample["eyes"] = new_eyes

    sample["box"] = (0, 0, x2 - x1, y2 - y1)

    return sample