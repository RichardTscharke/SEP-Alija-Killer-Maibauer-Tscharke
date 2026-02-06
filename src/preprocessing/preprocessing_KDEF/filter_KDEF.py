class KDEFfilter:

    def __init__(self, parameters, base_amount):
        """
        Encapsulates emotion- and pose-based filtering rules for KDEF.
        """

        # Defines the rules for each emotion class
        self.rules = {}

        for emotion, (positions, ratio) in parameters.items():
            
            self.rules[emotion] = {
                # Allowed head positions
                "positions": set(positions),
                # Final amount of images after downsampling
                "target": int(base_amount * ratio * len(positions))
            }

    # Called by sort_KDEF to check rules for each emotion class
    def allows(self, emotion, position):
        return (
            emotion in self.rules and
            position in self.rules[emotion]["positions"]
        )




