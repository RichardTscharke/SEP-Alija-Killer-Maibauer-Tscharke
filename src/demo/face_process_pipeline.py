from preprocessing.aligning.pipeline import preprocess_image
from preprocessing.detectors.retinaface import RetinaFaceDetector
from demo.draw_box_landmarks import draw_box_landmarks


class FaceStreamProcessor:

    def __init__(self, tracker, detect_every_n = 5):
        self.tracker = tracker
        self.detector = RetinaFaceDetector(device="cpu")
        self.detect_every_n = detect_every_n
        self.frame_idx = 0
        self.last_face = None

    def process_frame(self, frame):
        self.frame_idx += 1

        do_detect = (
            self.frame_idx % self.detect_every_n == 0
            or self.last_face is None
            or not self.tracker.active
        )

        # for every n-th frame
        if do_detect: 
            faces = self.detector.detect_face(frame)
            if faces:
                faces = sorted(faces, key = lambda f: f["confidence"], reverse = True)
                self.last_face = faces[0]
                self.tracker.init(frame, self.last_face["box"])

        # for all frames between n+1 and 2n-1 frames
        elif self.tracker.active:
            ok, box = self.tracker.update(frame)

            if ok and self.last_face is not None:
                self.last_face["box"] = box
                self.last_face["original"]["box"] = box

            else:
                self.tracker.reset()
                self.last_face = None

        sample = None
        if self.last_face is not None:
            sample = {
                "image": frame,
                "box": self.last_face["box"],
                "eyes": self.last_face["eyes"],
                "original": self.last_face["original"],
            }

            sample = preprocess_image(
                sample,
                do_clipping=True,
                do_cropping=True,
                do_aligning=True
            )

        outputs = {"webcam": frame}

        if sample is not None:
            #outputs["aligned_face"] = sample["image"] Uncomment if you want aligned picture in the demo.
            draw_box_landmarks(frame, sample["original"])

        return outputs
