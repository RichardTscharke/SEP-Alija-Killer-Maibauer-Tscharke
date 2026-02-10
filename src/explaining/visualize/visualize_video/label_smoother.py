class LabelSmoother:

    def __init__(self, alpha=0.3, every_nth_frame=5):
        self.alpha = alpha
        self.every_nth_frame=every_nth_frame
        self.state = None

    def __call__(self, probs, frame_idx):

        if self.state is None:
            self.state = probs
            return probs
        
        if frame_idx % self.every_nth_frame == 0:
            self.state = (
                self.alpha * probs + (1 - self.alpha) * self.state
            )

        return self.state