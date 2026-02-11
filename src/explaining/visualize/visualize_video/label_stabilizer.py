class LabelStabilizer:
    '''
    Converts smoothed class probabilities into a discrete label decision.
    Acts as a confidence gate rather than suppression tool due to weak classes.
    '''
    def __init__(self, min_conf = 0.6):

        # Minimum confidence a prediction must have in order to be displayed
        self.min_conf = min_conf

    def __call__(self, probs):
        
        # Determine top 2 highest predictions
        top2 = sorted(enumerate(probs),
                      key=lambda x: x[1],
                      reverse=True)[:2]
        
        (idx1, conf1), (idx2, conf2) = top2

        # No prediction higher than min_conf exists
        if conf1 < self.min_conf:
            return [None, None]
        
        # No second prediciton higher than min_conf exists
        elif conf2 < self.min_conf:
            return [(idx1, conf1), None]

        # Two top candidates exist
        else:
            return [(idx1, conf1), (idx2, conf2)]