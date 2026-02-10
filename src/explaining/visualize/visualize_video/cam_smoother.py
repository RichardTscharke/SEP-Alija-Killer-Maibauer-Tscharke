class CamSmoother:

    def __init__(self, alpha=0.2, every_nth_frame=5):
        self.alpha = alpha
        self.nth_frame = every_nth_frame
        self.prev_cam = None

    def __call__(self, cam, frame_idx):
        
        # Initialization
        if self.prev_cam is None:
            self.prev_cam = cam
            return cam
        
        if frame_idx % self.nth_frame == 0:
            self.prev_cam = (
                self.alpha * cam + (1 - self.alpha) * self.prev_cam
            )

        return self.prev_cam