import copy
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from . import actions
from .common.preprocessor import Preprocessor

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
        Convert observations and states into actions to take
        :param states: list of environment states to process
        :return: tuple of actions, states
        """

        raise NotImplementedError



class DQNAgent (BaseAgent):
    """
    DQNAgent is a memoryless DQN agent which calculates Q values
    from the observations and  converts them into the actions using action_selector
    """
    def __init__ (self, model: nn.Module, action_selector: actions.ActionSelector, device="cpu", preprocessor=Preprocessor.default_tensor):
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
        actions = self.action_selector(q)

        return actions

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
    Policy agent gets action probabilities from the model and samples actions from it
    """
    def __init__ (self, model, action_selector=actions.ProbabilityActionSelector(), device="cpu", apply_softmax=False, preprocessor=Preprocessor.default_tensor):
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
