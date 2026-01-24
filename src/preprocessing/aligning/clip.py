def clip_face(sample, clip_ratio = 0.4):

    if not isinstance(clip_ratio, (int, float)) or clip_ratio <= 0:
        return None

    image = sample["image"]
    x, y, w, h = sample["box"]

    h_img, w_img = image.shape[:2]

    w_clip = int(w * clip_ratio)
    h_clip = int(h * clip_ratio)

    x1 = max(0, x - w_clip)
    y1 = max(0, y - h_clip)

    x2 = min(w_img, x + w + w_clip)
    y2 = min(h_img, y + h + h_clip)

    if x2 <= x1 or y2 <= y1:
        return None
    
    sample["box"] = (x1, y1, x2 - x1, y2 - y1)

    return sample