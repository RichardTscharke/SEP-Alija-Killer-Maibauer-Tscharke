import threading
import time

class FERWorker:
    """
    Asynchronous inference worker for live FER demo.
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
        self.preprocess_f = preprocess_f

        # How often detection + inference is executed
        self.detect_every_n = max(1, int(detect_every_n))
        self.frame_counter = 0

        # Injected inference function (XAI or no XAI)
        self.infer_f = infer_f

        self.latest_frame = None

        # Last computed inference
        self.last_result = {
            "sample" : None,
            "probs"  : None,
            "cam"    : None,
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
        Pipeline per iteration:
        - Get newest frame
        - Detect face
        - Preprocess face
        - Run inference
        - Store results
        '''

        while self.running:

            frame = None

            # Drop older frames and always process only the newest (minimize latency)
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame
                    self.latest_frame = None

            # If no frame available, wait a bit to reduce CPU load
            if frame is None:
                time.sleep(0.001)
                continue

            # Run detection + inference only every n-th processed frame
            self.frame_counter += 1
            if self.frame_counter % self.detect_every_n != 0:
                continue

            # Face Detection (if no face => delete overlays)
            sample = self.preprocess_f(frame)

            if sample is None:
                with self.lock:
                    self.last_result = {"sample": None, "probs": None, "cam": None}
                continue

            # Inference (XAI or no XAI) -> try-except due to thread risks
            try:
                result = self.infer_f(sample)

            except Exception as e:
                print(f"[FERWorker] {e}")
                print(f"[FERWorker] {type(e)}: {e}")
                raise

            # Store full sample to allow renderer access to metadata
            with self.lock:
                self.last_result = {
                    "sample": sample,
                    "probs": result["probs"],
                    "cam": result["cam"]
                }

