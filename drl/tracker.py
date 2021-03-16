import math

from typing import List

from . import actions
from . import experience

class Tracker:
    def __init__ (self, start_val: float, end_val: float, steps: int, lin: bool=True):
        self.start_val = start_val
        self.end_val = end_val
        self.steps = steps
        self.lin = lin
        self.val = self.start_val

    def update (self, step: int):
        if self.lin:
            self._lin_update(step)
        else:
            self._exp_update(step)

    def _lin_update (self, step: int):
        self.val = self.start_val - (self.start_val - self.end_val) * step / self.steps

    def _exp_update (self, step: int):
        self.val = self.start_val - (self.start_val - self.end_val) * math.log(step, self.steps)

class EpsilonTracker (Tracker):
    def __init__ (self, selector: actions.EpsilonGreedySelector, start_val: float, end_val: float, steps: int, lin: bool=True):
        assert isinstance(selector, actions.EpsilonGreedySelector)
        super(EpsilonTracker, self).__init__(start_val, end_val, steps, lin)
        self.selector = selector
        self.selector.epsilon = self.val

    def update (self, step: int):
        super(EpsilonTracker, self).update(step)
        self.selector.epsilon = self.val

class BetaTracker (Tracker):
    def __init__ (self, p_buffer: experience.PrioReplayBuffer, start_val: float, end_val: float, steps: int, lin: bool=True):
        assert(p_buffer, experience.PrioReplayBuffer)
        super(BetaTracker, self).__init__(start_val, end_val, steps, lin)
        self.p_buffer = p_buffer

    def update (self, step: int):
        super(BetaTracker, self).update(step)
        self.p_buffer.beta = self.val
