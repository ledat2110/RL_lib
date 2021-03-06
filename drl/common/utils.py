import sys
import time
import operator
from datetime import timedelta
import numpy as np
import collections
import math

import torch
import torch.nn as nn

def cal_cont_logprob (mu_v: torch.Tensor, logstd_v: torch.Tensor, actions_v: torch.Tensor):
    assert isinstance(mu_v, torch.Tensor)
    assert isinstance(logstd_v, torch.Tensor)
    assert isinstance(actions_v, torch.Tensor)
    p1 = - ((mu_v - actions_v) ** 2) / (2 * torch.exp(logstd_v).clamp(min=1e-3))
    p2 = - torch.log(torch.sqrt(2 * math.pi * torch.exp(logstd_v)))

    return p1 + p2

def get_conv_out (conv_net, input_shape):
    o = conv_net(torch.zeros(1, *input_shape))
    return int(np.prod(o.size()))

class SMAQueue:
    """
    Queue of fixed size with mean, max, min operations
    """
    def __init__(self, size):
        self.queue = collections.deque()
        self.size = size

    def __iadd__(self, other):
        if isinstance(other, (list, tuple)):
            self.queue.extend(other)
        else:
            self.queue.append(other)
        while len(self.queue) > self.size:
            self.queue.popleft()
        return self

    def __len__(self):
        return len(self.queue)

    def __repr__(self):
        return "SMAQueue(size=%d)" % self.size

    def __str__(self):
        return "SMAQueue(size=%d, len=%d)" % (self.size, len(self.queue))

    def min(self):
        if not self.queue:
            return None
        return np.min(self.queue)

    def mean(self):
        if not self.queue:
            return None
        return np.mean(self.queue)

    def max(self):
        if not self.queue:
            return None
        return np.max(self.queue)


class SpeedMonitor:
    def __init__(self, batch_size, autostart=True):
        self.batch_size = batch_size
        self.start_ts = None
        self.batches = None
        if autostart:
            self.reset()

    def epoch(self):
        if self.epoches is not None:
            self.epoches += 1

    def batch(self):
        if self.batches is not None:
            self.batches += 1

    def reset(self):
        self.start_ts = time.time()
        self.batches = 0
        self.epoches = 0

    def seconds(self):
        """
        Seconds since last reset
        :return:
        """
        return time.time() - self.start_ts

    def samples_per_sec(self):
        """
        Calculate samples per second since last reset() call
        :return: float count samples per second or None if not started
        """
        if self.start_ts is None:
            return None
        secs = self.seconds()
        if abs(secs) < 1e-5:
            return 0.0
        return (self.batches + 1) * self.batch_size / secs

    def epoch_time(self):
        """
        Calculate average epoch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.epoches > 0:
            s /= self.epoches + 1
        return timedelta(seconds=s)

    def batch_time(self):
        """
        Calculate average batch time
        :return: timedelta object
        """
        if self.start_ts is None:
            return None
        s = self.seconds()
        if self.batches > 0:
            s /= self.batches + 1
        return timedelta(seconds=s)


class WeightedMSELoss(nn.Module):
    def __init__(self, size_average=True):
        super(WeightedMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, input, target, weights=None):
        if weights is None:
            return nn.MSELoss(self.size_average)(input, target)

        loss_rows = (input - target) ** 2
        if len(loss_rows.size()) != 1:
            loss_rows = torch.sum(loss_rows, dim=1)
        res = (weights * loss_rows).sum()
        if self.size_average:
            res /= len(weights)
        return res


class SegmentTree(object):
    def __init__(self, capacity, operation, neutral_element):
        """Build a Segment Tree data structure.

        https://en.wikipedia.org/wiki/Segment_tree

        Can be used as regular array, but with two
        important differences:

            a) setting item's value is slightly slower.
               It is O(lg capacity) instead of O(1).
            b) user has access to an efficient `reduce`
               operation which reduces `operation` over
               a contiguous subsequence of items in the
               array.

        Paramters
        ---------
        capacity: int
            Total size of the array - must be a power of two.
        operation: lambda obj, obj -> obj
            and operation for combining elements (eg. sum, max)
            must for a mathematical group together with the set of
            possible values for array elements.
        neutral_element: obj
            neutral element for the operation above. eg. float('-inf')
            for max and 0 for sum.
        """
        assert capacity > 0 and capacity & (capacity - 1) == 0, "capacity must be positive and a power of 2."
        self._capacity = capacity
        self._value = [neutral_element for _ in range(2 * capacity)]
        self._operation = operation

    def _reduce_helper(self, start, end, node, node_start, node_end):
        if start == node_start and end == node_end:
            return self._value[node]
        mid = (node_start + node_end) // 2
        if end <= mid:
            return self._reduce_helper(start, end, 2 * node, node_start, mid)
        else:
            if mid + 1 <= start:
                return self._reduce_helper(start, end, 2 * node + 1, mid + 1, node_end)
            else:
                return self._operation(
                    self._reduce_helper(start, mid, 2 * node, node_start, mid),
                    self._reduce_helper(mid + 1, end, 2 * node + 1, mid + 1, node_end)
                )

    def reduce(self, start=0, end=None):
        """Returns result of applying `self.operation`
        to a contiguous subsequence of the array.

            self.operation(arr[start], operation(arr[start+1], operation(... arr[end])))

        Parameters
        ----------
        start: int
            beginning of the subsequence
        end: int
            end of the subsequences

        Returns
        -------
        reduced: obj
            result of reducing self.operation over the specified range of array elements.
        """
        if end is None:
            end = self._capacity
        if end < 0:
            end += self._capacity
        end -= 1
        return self._reduce_helper(start, end, 1, 0, self._capacity - 1)

    def __setitem__(self, idx, val):
        # index of the leaf
        idx += self._capacity
        self._value[idx] = val
        idx //= 2
        while idx >= 1:
            self._value[idx] = self._operation(
                self._value[2 * idx],
                self._value[2 * idx + 1]
            )
            idx //= 2

    def __getitem__(self, idx):
        assert 0 <= idx < self._capacity
        return self._value[self._capacity + idx]


class SumSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(SumSegmentTree, self).__init__(
            capacity=capacity,
            operation=operator.add,
            neutral_element=0.0
        )

    def sum(self, start=0, end=None):
        """Returns arr[start] + ... + arr[end]"""
        return super(SumSegmentTree, self).reduce(start, end)

    def find_prefixsum_idx(self, prefixsum):
        """Find the highest index `i` in the array such that
            sum(arr[0] + arr[1] + ... + arr[i - i]) <= prefixsum

        if array values are probabilities, this function
        allows to sample indexes according to the discrete
        probability efficiently.

        Parameters
        ----------
        perfixsum: float
            upperbound on the sum of array prefix

        Returns
        -------
        idx: int
            highest index satisfying the prefixsum constraint
        """
        assert 0 <= prefixsum <= self.sum() + 1e-5
        idx = 1
        while idx < self._capacity:  # while non-leaf
            if self._value[2 * idx] > prefixsum:
                idx = 2 * idx
            else:
                prefixsum -= self._value[2 * idx]
                idx = 2 * idx + 1
        return idx - self._capacity


class MinSegmentTree(SegmentTree):
    def __init__(self, capacity):
        super(MinSegmentTree, self).__init__(
            capacity=capacity,
            operation=min,
            neutral_element=float('inf')
        )

    def min(self, start=0, end=None):
        """Returns min(arr[start], ...,  arr[end])"""

        return super(MinSegmentTree, self).reduce(start, end)

class MeanBuffer:
    def __init__ (self, capacity: int):
        assert isinstance(capacity, int)
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
        self.sum = 0.0

    def add (self, val: float):
        if len(self.buffer) == self.capacity:
            self.sum -= self.buffer[0]
        self.buffer.append(val)
        self.sum += val

    def mean (self):
        if not self.buffer:
            return 0.0
        return self.sum / len(self.buffer)

    def __len__ (self):
        return len(self.buffer)

    @property
    def size (self):
        return self.capacity
