import threading
import time


class FERWorker:
    """
    Runs face detection + FER in a background thread.
    Stores latest results for non-blocking access.
    """

    def __init__(self, detector, model, device, detect_every_n,
                 preprocess_f,
                 to_tensor_f,
                 run_model_f):

        self.detector = detector
        self.model    = model
        self.device   = device

        self.detect_every_n = detect_every_n
        self.frame_counter  = 0

        self.preprocess_f = preprocess_f
        self.to_tensor_f  = to_tensor_f
        self.run_model_f  = run_model_f

        self.latest_frame = None
        self.last_face    = None
        self.last_probs   = None

        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(
            target=self.worker_loop,
            daemon=True
        )
        self.thread.start()


    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame


    def get_results(self):
        with self.lock:
            return self.last_face, self.last_probs

    def stop(self):
        self.running = False


    def worker_loop(self):

        while self.running:

            frame = None

            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                    self.latest_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            self.frame_counter += 1

            if self.frame_counter % self.detect_every_n != 0:
                time.sleep(0.001)
                continue

            faces = self.detector.detect_face(frame)

            if not faces:
                time.sleep(0.002)
                continue

            faces = sorted(faces, key=lambda f: f["confidence"], reverse=True)
            face = faces[0]

            sample = self.preprocess_f(frame, face)

            if sample is None:
                continue

            tensor = self.to_tensor_f(sample, self.device)
            _, probs = self.run_model_f(self.model, tensor)

            with self.lock:
                self.last_face = face
                self.last_probs = probs.squeeze().cpu().numpy()