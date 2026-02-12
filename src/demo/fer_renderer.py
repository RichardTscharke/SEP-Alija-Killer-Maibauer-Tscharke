import cv2

from explaining.visualize.visualize_video.overlay import (
    overlay_gradcam,
    insert_emotion_label,
    draw_box_landmarks
)


class FERRenderer:
    """
    Handles all visiual overlays for the demo:
    - Grad-CAM heatmap
    - Face box + landmarks
    - Emotion label
    - Status overlay
    """

    def __init__(
        self,
        threshold,
        cam_smoother,
        label_smoother,
        label_stabilizer,
    ):
        # Confidence threshold for activation signals
        self.threshold = threshold

        # Temporal smoothing utilities
        self.cam_smoother = cam_smoother
        self.label_smoother = label_smoother
        self.label_stabilizer = label_stabilizer


    def render(
        self,
        frame,
        result,
        frame_idx,
        detect_every_n,
        show_landmarks,
        enable_xai,
    ):
        '''
        Applies all visiual overlays to the current frame.
        '''

        # If no inference result avilable, return original frame
        if result is None:
            return frame

        # Extract inference outputs
        sample = result.get("sample")
        probs = result.get("probs")
        cam = result.get("cam")

        # Grad-CAM Heatmap (if available)
        if enable_xai and cam is not None: 
            cam = self.cam_smoother(cam, frame_idx)
            frame = overlay_gradcam(frame, cam, self.threshold)
        
        # Face bounding box and landmarks
        if show_landmarks and sample is not None:
            frame = draw_box_landmarks(frame, sample["original"])

        # Emotion label
        if probs is not None:

            # Smooth probability distribution
            probs = self.label_smoother(probs, frame_idx)

            # Select top 2 confident predictions
            top2 = self.label_stabilizer(probs)

            # Render label
            frame = insert_emotion_label(frame, top2)
        
        # Status overlay 
        frame = self.draw_status_overlay(
            frame,
            detect_every_n,
            show_landmarks,
            enable_xai,
        )

        return frame


    def draw_status_overlay(
        self,
        frame,
        detect_every_n,
        show_landmarks,
        enable_xai,
    ):
        '''
        Draws a small system status box in the top-right corner.
        '''

        # Status text string
        text = (
            f"detect_every: {detect_every_n} | "
            f"Keypoints ('k') : {'ON' if show_landmarks else 'OFF'} | "
            f"Heatmap ('h'): {'ON' if enable_xai else 'OFF'}"
        )

        # Get frame width for alignment
        _, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        padding = 10

        # Compute text bounding box
        (text_w, text_h), _ = cv2.getTextSize(
            text, font, font_scale, thickness
        )

        # Background rectangle coordinates
        x1 = w - text_w - 2 * padding
        y1 = 10
        x2 = w - 10
        y2 = 10 + text_h + 2 * padding

        # Draw background rectangle
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Draw text
        cv2.putText(
            frame,
            text,
            (x1 + padding, y2 - padding),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )

        return frame
