import numpy as np
import collections

class MeanBuffer:
    def __init__ (self, capacity: int):
        assert isinstance(capacity, int)
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add (self, val: float):
        if len(self.buffer) == self.capacity:
            self.sum -= self.buffer[0]
        self.buffer.append(val)
        self.sum += val

    def mean (self):
        if not self.buffer:
            return 0.0
        return self.sum / len(self.buffer)

    def __len__ (self):
        return len(self.buffer)

    @property
    def size (self):
        return self.capacity
