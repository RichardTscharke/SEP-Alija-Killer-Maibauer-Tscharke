from src.explaining.explain_utils import preprocess_frame
from src.demo.fer_worker import FERWorker
from src.demo.fer_renderer import FERRenderer
from src.demo.infer_frame import infer_frame


class FERStreamController:
    '''
    High-level controller for the demo.
    Pipeline:
    - Hold UI state (XAI?, landmarks?)
    - Connect worker with renderer
    - Provide process_frame callback for WebCam
    '''

    def __init__(
            self,
            device,
            model,
            target_layer,
            detector,
            detect_every_n,
            threshold,
            cam_smoother,
            label_smoother,
            label_stabilizer,
            run_model_f,
    ):
        
        self.device = device
        
        # UI utils
        self.enable_xai = True
        self.show_landmarks = True
        self.frame_idx = 0

        # FERRenderer handles the overlays per frame
        self.renderer = FERRenderer(
            threshold,
            cam_smoother,
            label_smoother,
            label_stabilizer
        )

        # FerWorker runs detection + preprocessing + inference (async)
        self.worker = FERWorker(
            detector,
            model,
            device,
            detect_every_n,
            self.preprocess_f,
            self.build_infer_adapter(
                model,
                target_layer,
                run_model_f
            )
        )

    # Adapter injects static inference dependencies (model, layer)
    # and dynamic XAI state (enable_xai) into infer_frame
    def build_infer_adapter(self, model, target_layer, run_model_f):

        def infer(sample):

            return infer_frame(
                sample,
                model,
                target_layer,
                self.enable_xai,
                run_model_f
            )
        
        return infer

    # Our preprocessing pipeline: Clip -> Crop -> Align -> Tensor
    def preprocess_f(self, frame, face=None):

        # Build and return sample directory from face
        return preprocess_frame(frame, self.worker.detector, self.device)

    # Enable or disable Grad-CAM heatmap calculation + overlay
    def toggle_xai(self):
        self.enable_xai = not self.enable_xai

    # Enable or disable bounding box and landmark visualization
    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks

    # Adjust detection frequency 
    def set_detect_every_n(self, n):
        self.worker.set_detect_every_n(n)


    # Main callback for Webcam
    def process_frame(self, frame):
        
        self.frame_idx += 1

        # Push latest frame to async worker (non-blocking)
        self.worker.update_frame(frame)

        # Retrieve most recent inference result (may be from older frame)
        result = self.worker.get_results()

        # Update renderer
        return self.renderer.render(
            frame,
            result,
            self.frame_idx,
            self.worker.detect_every_n,
            self.show_landmarks,
            self.enable_xai
        )

    def stop(self):
        self.worker.stop()



        

