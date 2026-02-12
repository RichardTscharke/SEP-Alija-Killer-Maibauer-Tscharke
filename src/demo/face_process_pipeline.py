import time
import cv2

from demo.fer_worker import FERWorker
from explaining.explain_utils import run_model, np_to_tensor
from explaining.visualize.visualize_video.overlay import overlay_gradcam, insert_emotion_label
from explaining.visualize.visualize_video.cam_smoother import CamSmoother
from explaining.visualize.visualize_video.label_smoother import LabelSmoother
from explaining.visualize.visualize_video.label_stabilizer import LabelStabilizer
from preprocessing.aligning.detect import build_sample_from_face
from preprocessing.aligning.pipeline import preprocess_image

from .draw_box_landmarks import draw_box_landmarks
from .fer_per_frame import draw_emotion_probs



class FaceStreamProcessor:

    def __init__(self, model, detector, detect_every_n, target_layer):

        self.cam_smoother = CamSmoother(alpha=0.2, every_nth_frame=1)
        self.label_smoother = LabelSmoother(alpha=0.3, every_nth_frame=5)
        self.label_stabilizer = LabelStabilizer(min_conf=0.3)

        self.frame_idx = 0
        self.threshold = 0.3

        self.enable_xai = False
        self.show_landmarks = True

        self.prev_time = time.time()
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5
        self.fps = 0.0
        self.frame_time_ms = 0.0
        self.fps_alpha = 0.1

        self.worker = FERWorker(
            detector=detector,
            model=model,
            device=next(model.parameters()).device,
            detect_every_n=detect_every_n,
            target_layer=target_layer,
            preprocess_f=self.preprocess,
            to_tensor_f=self.to_tensor,
            run_model_f=run_model
        )


    def preprocess(self, frame, face):
        sample = build_sample_from_face(frame, face)
        sample = preprocess_image(sample)

        if sample is None:
            return None
        
        sample["original_img"] = frame
        sample["input_tensor"] = self.to_tensor(sample, self.worker.device)

        return sample

    def to_tensor(self, sample, device):
        return np_to_tensor(sample["image"], device)

    def toggle_xai(self):
        self.enable_xai = not self.enable_xai
        self.worker.set_xai(self.enable_xai)

    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks

    def set_detect_every_n(self, n):
        self.worker.set_detect_every_n(n)

    def process_frame(self, frame):

        start_time = time.time()

        self.frame_idx += 1

        self.worker.update_frame(frame)

        face, probs, cam = self.worker.get_results()

        if cam is not None:
            cam = self.cam_smoother(cam, self.frame_idx)

            frame = overlay_gradcam(
                image=frame,
                cam=cam,
                threshold=self.threshold
            )
        
        if probs is not None:
            probs = self.label_smoother(probs, self.frame_idx)
            top2 = self.label_stabilizer(probs)
            frame = insert_emotion_label(frame, top2)

        if face is not None and self.show_landmarks:
            frame = draw_box_landmarks(frame, face["original"])
        
        '''
        if probs is not None:
            draw_emotion_probs(frame, probs)
        '''

        end_time = time.time()

        # Frame-Zeit immer aktuell
        self.frame_time_ms = (end_time - start_time) * 1000

        instant_fps = 1.0 / (end_time - self.prev_time)
        self.prev_time = end_time

        # FPS nur periodisch aktualisieren
        if end_time - self.last_fps_update > self.fps_update_interval:
            self.fps = (1 - self.fps_alpha) * self.fps + self.fps_alpha * instant_fps
            self.last_fps_update = end_time


        self.draw_status_overlay(frame)

        return frame


    def stop(self):
        self.worker.stop()


    def draw_status_overlay(self, frame):

        text = (
            f"FPS: {self.fps:.1f} | "
            f"{self.frame_time_ms:.1f} ms | "
            f"detect_every: {self.worker.detect_every_n} | "
            f"Landmarks: {'ON' if self.show_landmarks else 'OFF'} | "
            f"XAI: {'ON' if self.enable_xai else 'OFF'}"
        )

        h, w = frame.shape[:2]

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

        padding = 10
        x1 = w - text_w - 2 * padding
        y1 = 10
        x2 = w - 10
        y2 = 10 + text_h + 2 * padding

        # Hintergrundbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), -1)

        # Text
        cv2.putText(
            frame,
            text,
            (x1 + padding, y2 - padding),
            font,
            font_scale,
            (255, 255, 255),
            thickness
        )



        

