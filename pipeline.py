import numpy as np


class ProcessingPipeline:
    """
    pipeline
    """

    # TODO : refine inputs
    def __init__(self, len_of_targets: int, max_n: int):
        self.L = len_of_targets
        self.M = max_n

    def get_batches(self):
        batches = []
        slices = self._slices()
        for i in range(len(slices) - 1):
            batches.append((slices[i], slices[i + 1]))
        return batches

    def _slices(self):
        slices = np.arange(0, self.L, self.M)
        if slices[-1] != self.L:
            slices = np.append(slices, self.L)
        return slices
