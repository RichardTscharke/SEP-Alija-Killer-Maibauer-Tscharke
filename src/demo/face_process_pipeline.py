from demo.fer_worker import FERWorker
from explaining.explain_utils import run_model, np_to_tensor
from preprocessing.aligning.detect import build_sample_from_face
from preprocessing.aligning.pipeline import preprocess_image

from .draw_box_landmarks import draw_box_landmarks
from .fer_per_frame import draw_emotion_probs


class FaceStreamProcessor:

    def __init__(self, model, detector, detect_every_n):

        self.worker = FERWorker(
            detector=detector,
            model=model,
            device=next(model.parameters()).device,
            detect_every_n=detect_every_n,
            preprocess_f=self.preprocess,
            to_tensor_f=self.to_tensor,
            run_model_f=run_model
        )


    def preprocess(self, frame, face):
        sample = build_sample_from_face(frame, face)
        return preprocess_image(sample)

    def to_tensor(self, sample, device):
        return np_to_tensor(sample["image"], device)


    def process_frame(self, frame):

        self.worker.update_frame(frame)

        face, probs = self.worker.get_results()

        if face is not None:
            draw_box_landmarks(frame, face["original"])

        if probs is not None:
            draw_emotion_probs(frame, probs)

        return frame


    def stop(self):
        self.worker.stop()

        

