import cv2

class FaceTracker:
    def __init__(self):
        self.tracker = None
        self.active = False

    def init(self, frame, box):

        self.tracker = cv2.legacy.TrackerKCF_create() # cv2 versions < 4.5 use cv2.TrackerKCF_create()
        self.tracker.init(frame, tuple(box))
        self.active = True

    def update(self, frame): # returns (ok, box)

        if not self.active:
            return False, None
        
        ok, box = self.tracker.update(frame)
        if not ok:
            self.active = False
            return False, None
        
        return True, tuple(map(int, box))
    
    def reset(self):
        self.tracker = None
        self.active = False
