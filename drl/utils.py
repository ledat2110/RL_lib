import torch
import torch.nn as nn

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

class Preprocessor:
    @staticmethod
    def default_tensor (states):
        state =  np.expand_dims(states, 0)
        return torch.tensor(state)

    @staticmethod
    def float32_tensor (states):
        state =  np.expand_dims(states, 0)
        return torch.tensor(state, dtype=torch.float32)

def get_conv_out_size (conv_net: nn.Module, input_shape):
    o = conv_net(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))
