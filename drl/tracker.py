import math
import time
import sys
import numpy as np

from tensorboardX import SummaryWriter
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
        self.val = max(self.val, self.end_val)

    def _exp_update (self, step: int):
        self.val = self.start_val - (self.start_val - self.end_val) * math.log(step, self.steps)
        self.val = max(self.val, self.end_val)

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

class RewardTracker:
    def __init__ (sefl, writer: SummaryWriter, stop_reward: float):
        assert isinstance(writer, SummaryWriter)
        self.writer = writer
        self.stop_reward = stop_reward

    def __enter__ (self):
        self.ts = time.time()
        self.eps_ts = time.time()
        self.ts_frame = 0
        self.total_reward = []
        return self

    def __exit__ (self):
        self.writer.close()

    def update (self, reward, frame, epsilon=None):
        self.total_reward.append(reward)
        m_reward = np.mean(self.total_reward[-100:])

        now = time.time()
        speed = (frame - self.ts_frame) / (self.eps_ts - now)
        elapsed = now - self.ts

        self.ts_frame = frame
        self.eps_ts = now

        print("Episode: %d, Reward: %6.3f, Speed: %6.3f, Epsilon: %s, Elapsed: %.3f" % (len(self.total_reward), m_reward, speed, epsilon, elapsed))
        sys.stdout.flush()

        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("speed", speed, frame)
        self.writer.add_scalar("reward", m_reward, frame)

        if m_reward > self.stop_reward:
            print("Solved in %d steps and %d episodes"%(frame, len(self.total_reward)))
            return True

        return False
