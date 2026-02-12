import threading
import time


class FERWorker:
    """
    Asynchonous inference worker for live FER demo.
    Pipeline:
    - Receive latest frame from main thread
    - Run detection every n frames
    - Preprocess detected face
    - Call injected inference function
    - Store latest inference results
    """

    def __init__(
            self,
            detector,
            model,
            device,
            detect_every_n,
            preprocess_f,
            infer_f
    ):

        self.detector = detector
        self.model = model
        self.device = device

        # How often detection + inference is executed
        self.detect_every_n = max(1, int(detect_every_n))
        self.frame_counter = 0

        # Injected inference function (XAI or no XAI)
        self.infer_f = infer_f

        self.latest_frame = None

        # Last computed inference
        self.last_results = {
            "face" : None,
            "probs": None,
            "cam": None,
        }

        # Thread utils
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self.worker_loop, daemon=True)

        self.thread.start()

    
    # Receive newest frame from UI thread
    def update_frame(self, frame):
        with self.lock:
            self.latest_frame = frame

    # Return latest inference results
    def get_results(self):
        with self.lock:
            return self.last_result.copy()

    # Update detection frequency
    def set_detect_every_n(self, n):
        self.detect_every_n = max(1, int(n))
        self.frame_counter = 0

    # Stop worker thread
    def stop(self):
        self.running = False
        self.thread.join()


    def worker_loop(self):
        '''
        Pipeine per iteration:
        - Get newest frame
        - Detect face
        - Preprocess face
        - Run inference
        - Store results
        '''

        while self.running:

            frame = None

            # get latest frame (if it exists)
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None

            # If no frame available, wait a bit to reduce CPU load
            if frame is None:
                time.sleep(0.005)
                continue

            # Detect every n
            self.frame_counter += 1
            if self.frame_counter % self.detect_every_n != 0:
                continue

            # Face Detection
            faces = self.detector.detect_face(frame)
            if not faces:
                continue

            # Face Selection
            faces = sorted(faces, key=lambda f: f["confidence"], reverse=True)
            face = face[0]

            # Our preprocessing pipeline: Clip -> Crop -> Align
            sample = self.preprocess_f(frame, face)

            if sample is None:
                continue

            # Inference (XAI or no XAI)
            result = self.infer_f(sample, self.model)

            # Store results (thread safe)
            with self.lock:
                self.last_result = {
                    "face": face,
                    "probs": result["probs"],
                    "cam": result["cam"]
                }

