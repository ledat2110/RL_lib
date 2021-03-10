import math
import torch
import numpy as np

class Preprocessor:
    @staticmethod
    def default_tensor (states):
        state =  np.expand_dims(states, 0)
        return torch.tensor(state)

    @staticmethod
    def float32_tensor (states):
        state =  np.expand_dims(states, 0)
        return torch.tensor(state, dtype=torch.float32)