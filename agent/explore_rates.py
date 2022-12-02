# The explore rate decays from 1 to 0.1 linearly over the frame 
# limit defined in the training_metadata and stays at 0.1 thereafter
class Decay_Explore_Rate:
    def get(self, step, step_limit):
        return max(0.1, (1 - float(step) / step_limit))

    def __str__(self):
        return 'max(0.1, (1 - float(training_metadata.frame) / training_metadata.frame_limit))'