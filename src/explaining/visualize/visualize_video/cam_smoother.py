class CamSmoother:
    '''
    Applies exponential moving average smoothing to Grad-CAM heatmaps.
    Goal: Reduce frame-to-frame flickering in heatmap overlay.
    '''

    def __init__(self, alpha=0.2, every_nth_frame=5):

        # Smoothing factor
        self.alpha = alpha

        # Apply smoothing update every n-th frame (default: every frame)
        self.nth_frame = every_nth_frame

        # Store previous smoothed CAM
        self.prev_cam = None

    def __call__(self, cam, frame_idx):
        
        # Initilaize with first CAM
        if self.prev_cam is None:
            self.prev_cam = cam
            return cam
        
        # Update smoothed CAM only every n-th frame
        if frame_idx % self.nth_frame == 0:
            self.prev_cam = (
                self.alpha * cam + (1 - self.alpha) * self.prev_cam
            )

        # Return last smoothed CAM
        return self.prev_cam