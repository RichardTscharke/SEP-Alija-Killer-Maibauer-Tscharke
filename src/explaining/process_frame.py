from .explain_utils import preprocess_frame
from .cam.explain_frame import explain_frame
from .visualize.visualize_video import overlay_gradcam

def process_frame(
        frame,
        detector,
        model,
        target_layer,
        device,
        cam_smoother,
        frame_idx,
        threshold
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

    return overlayed