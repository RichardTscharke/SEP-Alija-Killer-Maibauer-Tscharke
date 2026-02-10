class LabelStabilizer:
    
    def __init__(self, min_conf = 0.6):
        self.current_idx = None
        self.current_conf = None
        self.min_conf = min_conf

    def __call__(self, probs):
        idx = int(probs.argmax())
        conf = float(probs[idx])

        if conf < self.min_conf:
            return None, conf
        
        self.current_idx = idx
        self.current_conf = conf

        return self.current_idx, self.current_conf