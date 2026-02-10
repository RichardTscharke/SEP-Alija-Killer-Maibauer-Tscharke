def clip_face(sample, clip_ratio = 0.4):
    '''
    Expands the current bounding box by clip_ratio within image boundaries.
    The image and landmarks are NOT modified.
    '''
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
    
    # Enlarges the box around the original bounding box to reduce border artifacts 
    sample["box"] = (x1, y1, x2 - x1, y2 - y1)

    # Store the enlargened box + applied ratio since sample["box"] will be overwritten later
    sample["meta"]["clip_box"] = sample["box"]
    sample["meta"]["clip_ratio"] = clip_ratio
    
    return sample