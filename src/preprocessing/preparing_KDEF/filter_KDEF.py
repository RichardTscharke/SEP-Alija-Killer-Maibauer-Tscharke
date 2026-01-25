class KDEFfilter:

    def __init__(self, parameters, base_amount):

        self.cfg = {}

        for emotion, (positions, ratio) in parameters.items():
            
            self.cfg[emotion] = {
                "positions": set(positions),
                "target": int(base_amount * ratio * len(positions))
            }

    def allows(self, emotion, position):
        return (
            emotion in self.cfg and
            position in self.cfg[emotion]["positions"]
        )




