import math
import torch
import numpy as np

class Preprocessor:
    @staticmethod
    def default_tensor (states: np.ndarray):
        return torch.tensor([states])

    @staticmethod
    def float32_tensor (states: np.ndarray):
        return torch.tensor([states], dtype=torch.float32)