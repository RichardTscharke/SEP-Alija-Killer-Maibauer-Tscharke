from .explain_utils import preprocess_frame
from .cam.explain_frame import explain_frame
from .visualize.visualize_video.overlay import overlay_gradcam, insert_emotion_label


def process_frame(
        frame,
        detector,
        model,
        target_layer,
        device,
        cam_smoother,
        frame_idx,
        threshold,
        label_smoother,
        label_stabilizer
):
    sample = preprocess_frame(frame, detector, device)
    if sample is None:
        return frame
    
    sample = explain_frame(
        sample=sample,
        model=model,
        target_layer=target_layer,
    )

    cam = sample["cam_original"]
    cam = cam_smoother(cam, frame_idx)

    overlayed = overlay_gradcam(
        image=sample["original_img"],
        cam=cam,
        threshold=threshold
    )

    probs = label_smoother(sample["probs"], frame_idx)
    idx, conf = label_stabilizer(probs)

    labelized = insert_emotion_label(
        overlayed,
        idx,
        conf
    )

    return labelized