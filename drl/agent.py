import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from . import action
from .utils import Preprocessor

from typing import List

class BaseAgent:
    """
    Abstract Agent interface
    """
    def initial_state (self):
        """
        Should create initial empty state for the agent. It will be called for the start of the episode
        :return: Anything agent want to remember
        """
        return None

    def __call__ (self, state):
        """
        Convert observations and states into action to take
        :param states: list of environment states to process
        :return: tuple of action, states
        """

        raise NotImplementedError



class DQNAgent (BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the action using action_selector
    """
    def __init__ (self, model: nn.Module, action_selector: action.ActionSelector, device="cpu", preprocessor=Preprocessor.default_tensor):
        self.model = model
        self.action_selector = action_selector
        self.preprocessor = preprocessor
        self.device = device

    @torch.no_grad()
    def __call__ (self, state: np.ndarray) -> np.ndarray:
        if self.preprocessor is not None:
            state = self.preprocessor(state)
        if torch.is_tensor(state):
            state = state.to(self.device)
        
        q_v = self.model(state)
        q = q_v.squeeze(0).data.cpu().numpy()
        action = self.action_selector(q)

        return action

class TargetNet:
    """
    Wrapper around model which provides copy of it instead of trained weights
    """
    def __init__ (self, model: nn.Module):
        self.model = model
        self.target_model = copy.deepcopy(model)

    def sync (self):
        self.target_model.load_state_dict(self.model.state_dict())

    def alpha_sync (self, alpha: float):
        assert isinstance(alpha, float)
        assert 0.0 <= alpha <= 1.0
        state = self.model.state_dict()
        tgt_state = self.target_model.state_dict()
        for k, v in state.items():
            tgt_state[k] = tgt_state[k] * alpha + (1 - alpha) * v
        
        self.target_model.load_state_dict(tgt_state)

class PolicyAgent (BaseAgent):
    """
    Policy agent gets action probabilities from the model and samples action from it
    """
    def __init__ (self, model, action_selector=action.ProbabilityActionSelector(), device="cpu", apply_softmax=False, preprocessor=Preprocessor.default_tensor):
        self.model = model
        self.action_selector = action_selector
        self.device = device
        self.apply_softmax = apply_softmax
        self.preprocessor = preprocessor

    @torch.no_grad()
    def __call__ (self, state: List[np.ndarray]) -> np.ndarray:
        if self.preprocessor is not None:
            state = self.preprocessor(state)
        if torch.is_tensor(state):
            state = state.to(self.device)
        
        probs_v = self.model(state)
        if self.apply_softmax:
            probs_v = F.softmax(probs_v, dim=1)
        prob = probs_v.squeeze(0).data.cpu().numpy()
        action = self.action_selector(prob)

        return action

class ContA2CAgent (BaseAgent):
    def __init__ (self, net, device="cpu"):
        self.net = net
        self.device = device

    def __call__ (self, states):
        states_v = Preprocessor.float32_tensor(states)
        states_v = states_v.to(self.device)

        mu_v, var_v, _ = self.net(states_v)
        mu = mu_v.data.cpu().numpy()
        sigma = torch.sqrt(var_v).data.cpu().numpy()

        actions = np.random.normal(mu, sigma)
        return actions

class ThresholdAgent:
    def __init__ (self, eps: np.array, Q: np.array, action_dim: int):
        self.eps = eps
        self.Q = Q
        self.action_dim = action_dim
        self.production_flag = True

    def get_action (self, state: np.ndarray):
        action = np.zeros(self.action_dim, dtype=np.int32)
        action[0] = self.Q[0] if self.production_flag else 0

        for i in range(1, self.action_dim):
            if state[i] < self.eps[i]:
                action[i] = self.Q[i]

        return action

    def set_production_level (self, state: np.ndarray, num_storages: int):
        self.production_flag = False
        if (state[0] - np.sum(state[1:num_storages+1])) < self.eps[0]:
            self.production_flag = True
