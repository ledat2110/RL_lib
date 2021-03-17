import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math


class NoisyLinear (nn.Linear):
    def __init__ (self, in_features: int, out_features: int, sigma_init: float=0.017, bias: bool=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z = torch.zeros(out_features, in_features)
        self.register_buffer("epsilon_weight", z)
        
        if bias:
            w = torch.full((out_features, ), sigma_init)
            self.sigma_bias = nn.Parameter(w)
            z = torch.zeros(out_features)
            self.register_buffer("epsilon_bias", z)
        self.reset_parameters()

    def reset_parameters (self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward (self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
        v = self.sigma_weight * self.epsilon_weight.data + self.weight
        return F.linear(input, v, bias)

class NoisyFactorizedLinear (nn.Linear):
    def __init__ (self, in_features: int, out_features: int, sigma_zero: float=0.4, bias: bool=True):
        super(NoisyFactorizedLinear, self).__init__(in_features, out_features, bias)
        sigma_init = sigma_zero / math.sqrt(in_features)
        w = torch.full((out_features, in_features), sigma_init)
        self.sigma_weight = nn.Parameter(w)
        z1 = torch.zeros(1, in_features)
        self.register_buffer("epsilon_input", z1)
        z2 = torch.zeros(out_features, 1)
        self.register_buffer("epsilon_output", z2)
        
        if bias:
            w = torch.full((out_features, ), sigma_init)
            self.sigma_bias = nn.Parameter(w)

    def forward (self, input):
        self.epsilon_input.normal_()
        self.epsilon_output.normal_()

        func = lambda x: torch.sign(x) * torch.sqrt(torch.abs(x))
        eps_in = func(self.epsilon_input.data)
        eps_out = func(self.epsilon_output.data)

        bias = self.bias
        if bias is not None:
            bias = bias + self.sigma_bias * eps_out.t()

        noise_v = torch.mul(eps_in, eps_out)
        v = self.weight + self.sigma_weight * noise_v

        return F.linear(input, v, bias)


