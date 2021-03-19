import numpy as np


class ProcessingPipeline:
    """
    pipeline
    """

    def __init__(self, len_of_targets: int, max_n: int):
        '''
        :param len_of_targets: 当前需要batch的数据总长度
        :param max_n: 每个batch最多多少条数据
        '''
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
