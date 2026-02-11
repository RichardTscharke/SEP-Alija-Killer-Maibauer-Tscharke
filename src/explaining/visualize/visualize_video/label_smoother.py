class LabelSmoother:
    '''
    Applies exponential moving average smoothing to class probabilites.
    Goal: Reduce frame-to-frame flickering in prediction box.
    '''
    def __init__(self, alpha=0.3, every_nth_frame=5):

        # Smoothing factor
        self.alpha = alpha

        # Update state every n-th frame
        self.every_nth_frame=every_nth_frame

        # Store previous smoothed probabilities
        self.state = None

    def __call__(self, probs, frame_idx):

        # Initialize with first probability vector
        if self.state is None:
            self.state = probs
            return probs
        
        # Update smoothed probabilities only every n-th frame
        if frame_idx % self.every_nth_frame == 0:
            self.state = (
                self.alpha * probs + (1 - self.alpha) * self.state
            )

        # Return last smoothed probability vector
        return self.state