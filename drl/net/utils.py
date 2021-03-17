import torch
import torch.nn as nn

import numpy as np

def get_conv_out_size (conv_net: nn.Module, input_shape):
    o = conv_net(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))
