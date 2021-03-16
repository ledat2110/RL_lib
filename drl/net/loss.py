import torch
import torch.nn as nn

import numpy as np

from typing import List

from .. import agent

class Loss:
    def __init__ (self, device: str):
        self.device = device

    def __cal__ (self, batch: List[np.ndarray]):
        raise NotImplementedError

class DQNLoss (Loss):
    def __init__ (self, net: nn.Module, tgt_net: nn.Module, gamma: float, device: str):
        super(DQNLoss, self).__init__(device)
        self.net = net
        self.tgt_net = tgt_net
        self.gamma = gamma

    def __call__ (self, batch: List[np.ndarray]):
        states, actions, rewards, dones, next_states = batch

        states_v = torch.tensor(states, dtype=torch.float32).to(self.device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        actions_v = torch.tensor(actions).to(self.device)
        rewards_v = torch.tensor(rewards).to(self.device)
        done_mask = torch.BoolTensor(dones).to(self.device)

        sa_vals = self.net(states_v).gather(1, actions_v.unsqueeze(-1).type(torch.int64)).squeeze(-1)
        with torch.no_grad():
            next_s_vals = self.tgt_net(next_states_v).max(1)[0]
            next_s_vals[done_mask] = 0
            next_s_vals = next_s_vals.detach()

        expected_sa_vals = next_s_vals * self.gamma + rewards_v
        loss = nn.MSELoss(sa_vals, expected_sa_vals)

        return loss
