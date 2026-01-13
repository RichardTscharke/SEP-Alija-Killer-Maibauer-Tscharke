def clip_face(sample, clip_ratio = 0.4):

    image = sample["image"]
    x, y, w, h = sample["box"]

    h_img, w_img = image.shape[:2]

    w_clip = int(w * clip_ratio)
    h_clip = int(h * clip_ratio)

    x1 = max(0, x - w_clip)
    y1 = max(0, y - h_clip)

    x2 = min(w_img, x + w + w_clip)
    y2 = min(h_img, y + h + h_clip)

    sample["box"] = (x1, y1, x2 - x1, y2 - y1)

    return sample