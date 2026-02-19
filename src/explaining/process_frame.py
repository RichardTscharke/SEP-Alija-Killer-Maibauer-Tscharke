from src.explaining.explain_utils import preprocess_frame
from src.explaining.cam.explain_frame import explain_frame
from src.explaining.visualize.visualize_video.overlay import overlay_gradcam, insert_emotion_label


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
    '''
    Orchestrates XAI pipeline for a single video frame.
    Steps:
    - Face detection and alignment
    - Grad-CAM computation and backprojection
    - Temporal smoothing of CAM and predicitions
    - Blending frame with heatmap and emotion label insertion

    Returns the annotated frame.
    If no face is detected, returns the original frame unchanged.
    '''

    # Preprocessing pipeline (if no face detected this returns None)
    preprocessed = preprocess_frame(frame, detector, device)

    # Skip explanation if no valid face was detected
    if preprocessed is None:
        return frame
    
    # Run Grad-CAM explanation and add CAM + predictions to sample
    explained = explain_frame(
        sample=preprocessed,
        model=model,
        target_layer=target_layer,
    )

    # Retrieve CAM in original image coordinates
    cam = explained["cam_original"]

    # Apply temporal smoothing to stabilize CAM across frames
    cam = cam_smoother(cam, frame_idx)

    # Overlay CAM heatmap onto original frame
    overlayed = overlay_gradcam(
        image=explained["original_img"],
        cam=cam,
        threshold=threshold
    )

    # Smooth class probabilities temporally
    probs = label_smoother(explained["probs"], frame_idx)

    # Stabilize top 2 predicted labels to avoid flickering
    top2_pred = label_stabilizer(probs)

    # Insert emotion label and confidence into the frame
    labelized = insert_emotion_label(overlayed, top2_pred)

    return labelized