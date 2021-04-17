import math
import time
import sys
import numpy as np
import collections
import datetime

import torch

from functools import partial
from tensorboardX import SummaryWriter
from typing import List

from . import action
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
    def __init__ (self, selector: action.EpsilonGreedySelector, start_val: float, end_val: float, steps: int, lin: bool=True):
        assert isinstance(selector, action.EpsilonGreedySelector)
        super(EpsilonTracker, self).__init__(start_val, end_val, steps, lin)
        self.selector = selector
        self.selector.epsilon = self.val

    def update (self, step: int):
        super(EpsilonTracker, self).update(step)
        self.selector.epsilon = self.val

class BetaTracker (Tracker):
    def __init__ (self, p_buffer: experience.PrioReplayBufferNaive, start_val: float, end_val: float, steps: int, lin: bool=True):
        assert(p_buffer, experience.PrioReplayBufferNaive)
        super(BetaTracker, self).__init__(start_val, end_val, steps, lin)
        self.p_buffer = p_buffer

    def update (self, step: int):
        super(BetaTracker, self).update(step)
        self.p_buffer.beta = self.val

class TBMeanTracker:
    """
    TensorBoard value tracker: allows to batch fixed amount of historical values and write their mean into TB

    Designed and tested with pytorch-tensorboard in mind
    """
    def __init__(self, writer, batch_size):
        """
        :param writer: writer with close() and add_scalar() methods
        :param batch_size: integer size of batch to track
        """
        assert isinstance(batch_size, int)
        assert writer is not None
        self.writer = writer
        self.batch_size = batch_size

    def __enter__(self):
        self._batches = collections.defaultdict(list)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.writer.close()

    @staticmethod
    def _as_float(value):
        assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
        tensor_val = None
        if isinstance(value, torch.autograd.Variable):
            tensor_val = value.data
        elif torch.is_tensor(value):
            tensor_val = value

        if tensor_val is not None:
            return tensor_val.float().mean().item()
        elif isinstance(value, np.ndarray):
            return float(np.mean(value))
        else:
            return float(value)

    def track(self, param_name, value, iter_index):
        assert isinstance(param_name, str)
        assert isinstance(iter_index, int)

        data = self._batches[param_name]
        data.append(self._as_float(value))

        if len(data) >= self.batch_size:
            self.writer.add_scalar(param_name, np.mean(data), iter_index)
            data.clear()


class RewardTracker:
    def __init__(self, writer, min_ts_diff=1.0):
        """
        Constructs RewardTracker
        :param writer: writer to use for writing stats
        :param min_ts_diff: minimal time difference to track speed
        """
        self.writer = writer
        self.min_ts_diff = min_ts_diff

    def __enter__(self):
        self.ts = time.time()
        self.train_ts = time.time()
        self.ts_frame = 0
        self.total_rewards = []
        return self

    def __exit__(self, *args):
        self.writer.close()

    def reward(self, reward, frame, epsilon=None):
        self.total_rewards.append(reward)
        mean_reward = np.mean(self.total_rewards[-100:])
        ts_diff = time.time() - self.ts
        if ts_diff > self.min_ts_diff:
            speed = (frame - self.ts_frame) / ts_diff
            self.ts_frame = frame
            self.ts = time.time()
            epsilon_str = "None" if epsilon is None else "%.3f" % epsilon
            print("Frame: %d, Episode: %d, Reward: %6.3f, Speed: %6.3f, Epsilon: %s, Elapsed: %s" % (frame, len(self.total_rewards), mean_reward, speed, epsilon_str, str(datetime.timedelta(seconds=int(time.time() - self.train_ts)))))
            # print("%d: done %d episodes, mean reward %.3f, speed %.2f f/s%s" % (
            #     frame, len(self.total_rewards), mean_reward, speed, epsilon_str
            # ))
            sys.stdout.flush()
            self.writer.add_scalar("speed", speed, frame)
        if epsilon is not None:
            self.writer.add_scalar("epsilon", epsilon, frame)
        self.writer.add_scalar("reward_100", mean_reward, frame)
        self.writer.add_scalar("reward", reward, frame)
        return mean_reward if len(self.total_rewards) > 30 else None

# class RewardTracker:
#     def __init__ (self, writer: SummaryWriter, stop_reward: float):
#         assert isinstance(writer, SummaryWriter)
#         self.writer = writer
#         self.stop_reward = stop_reward

#     def __enter__ (self):
#         self.ts = time.time()
#         self.eps_ts = time.time()
#         self.ts_frame = 0
#         self.total_reward = []
#         return self

#     def __exit__ (self, exec_type, exec_val, exec_tb):
#         self.writer.close()

#     def update (self, reward, frame, epsilon=0):
#         self.total_reward.append(reward)
#         m_reward = np.mean(self.total_reward[-100:])

#         now = time.time()
#         speed = (frame - self.ts_frame) / (now - self.eps_ts)
#         elapsed = int(now - self.ts)

#         self.ts_frame = frame
#         self.eps_ts = now

#         print("Frame: %d, Episode: %d, Reward: %6.3f, Speed: %6.3f, Epsilon: %0.3f, Elapsed: %s" % (frame, len(self.total_reward), m_reward, speed, epsilon, str(datetime.timedelta(seconds=elapsed))))
#         sys.stdout.flush()

#         if epsilon is not None:
#             self.writer.add_scalar("epsilon", epsilon, frame)
#         self.writer.add_scalar("speed", speed, frame)
#         self.writer.add_scalar("reward", m_reward, frame)

#         if m_reward > self.stop_reward:
#             print("Solved in %d steps and %d episodes"%(frame, len(self.total_reward)))
#             return True

#         return False, m_reward

# class TBMeanTracker:
#     def __init__ (self, writer, mean_step: int):
#         assert isinstance (mean_step, int)
#         assert isinstance (writer, SummaryWriter)
#         self.writer = writer
#         self.m_step = mean_step

#     def __enter__ (self):
#         self._params = collections.defaultdict(partial(collections.deque, maxlen=self.m_step))
#         return self

#     def __exit__ (self, exec_type, exec_val, exec_tb):
#         self.writer.close()

#     @staticmethod
#     def _as_float(value):
#         assert isinstance(value, (float, int, np.ndarray, np.generic, torch.autograd.Variable)) or torch.is_tensor(value)
#         tensor_val = None
#         if isinstance(value, torch.autograd.Variable):
#             tensor_val = value.data
#         elif torch.is_tensor(value):
#             tensor_val = value

#         if tensor_val is not None:
#             return tensor_val.float().mean().item()
#         elif isinstance(value, np.ndarray):
#             return float(np.mean(value))
#         else:
#             return float(value)

#     def track (self, param_name: str, value, iter_idx: int, mean: bool=True):
#         assert isinstance(param_name, str)
#         assert isinstance(iter_idx, int)

#         data = self._params[param_name]
#         data.append(self._as_float(value))

#         if mean:
#             self.writer.add_scalar(param_name, np.mean(data), iter_idx)
#         else:
#             self.writer.add_scalar(param_name, data[-1], iter_idx)

#     def close (self):
#         self.writer.close()
