from .explain_utils import preprocess_frame
from .cam.explain_frame import explain_frame
from .visualize.visualize_video import overlay_gradcam

def process_frame(
        frame,
        detector,
        model,
        target_layer,
        device,
        threshold,
):
    sample = preprocess_frame(frame, detector, device)
    if sample is None:
        return None
    
    sample = explain_frame(
        sample=sample,
        model=model,
        target_layer=target_layer,
    )

    overlayed = overlay_gradcam(
        image=sample["original_img"],
        cam=sample["cam_original"],
        threshold=threshold
    )

    return overlayed