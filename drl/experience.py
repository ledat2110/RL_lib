import gym
import random
import collections

import numpy as np
import torch

from collections import namedtuple, deque
from typing import Tuple, List

from .agent import BaseAgent

Experience = namedtuple("Experience", ['state', 'action', 'reward', 'done', 'last_state'])
EpisodeEnded = namedtuple("EpisodeEnded", field_names=['reward', 'step'])

def unpack_data (exps: List[Experience]) -> Tuple:
    states, actions, rewards, dones, next_states = [], [], [], [], []
    for exp in exps:
        states.append(np.array(exp.state))
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.done)
        next_states.append(np.array(exp.last_state))

    states = np.array(states, copy=False)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    dones = np.array(dones, dtype=np.uint8)
    next_states = np.array(next_states, copy=False)
    
    return states, actions, rewards, dones, next_states

class ReplayBuffer:
    def __init__ (self, capacity: int):
        assert isinstance(capacity, int)
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=capacity)
    
    def __len__ (self):
        return len(self.buffer)

    def sample(self, batch_size: int):
        if len(self.buffer) <= batch_size:
            samples = self.buffer
        else:
            indices = np.random.choice(len(self.buffer), batch_size, replace=False)
            samples = [self.buffer[idx] for idx in indices]

        samples = unpack_data(samples)

        return samples
    
    def append (self, exp: Experience):
        self.buffer.append(exp)

    @property
    def size (self):
        return self.capacity

class PrioReplayBuffer:
    def __init__ (self, capacity: int, prob_alpha: float=0.6, beta: float=0.4):
        self.priorities = np.zeros((capacity, ), dtype=np.float32)
        self.buffer = []
        self.pos = 0
        self.prob_alpha = prob_alpha
        self.capacity = capacity
        self.beta = beta

    def __len__ (self):
        return len(self.buffer)

    def append(self, exp: Experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.pos] = exp
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int):
        total = len(self.buffer)
        probs = self.priorities[:total] ** self.prob_alpha
        probs /= probs.sum()

        if total <= batch_size:
            indices = np.array(range(total))
        else:
            indices = np.random.choice(total, batch_size, p=probs, replace=False)
        samples = [self.buffer[idx] for idx in indices]

        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = unpack_data(samples)

        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities (self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__ (self, env: gym.Env, agent: BaseAgent, buffer: ReplayBuffer=None, steps_count: int=2, gamma: float=0.99):
        """
        Create simple experience source
        :param env: environment to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, gym.Env)
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        self.agent = agent
        self.steps_count = steps_count
        self.gamma = gamma

        self.env = env
        self.buffer = buffer

        self.total_reward = None
        self.total_step = None
        self.reset()

    def reset (self):
        self.state = self.env.reset()
        self.cur_step = 0
        self.cur_reward = 0

    def play_steps (self):
        state = self.state
        actions = []
        rewards = []
        for _ in range(self.steps_count):
            action = self.agent(state)
            next_state, reward, done, _ = self.env.step(action)

            self.cur_reward += reward
            self.cur_step += 1

            rewards.append(reward)
            actions.append(action)

            if done:        
                next_state = state
                break

            state = next_state

        reward = 0
        for r in reversed(rewards):
            reward *= self.gamma
            reward += r

        exp = Experience(self.state, actions[0], reward, done, next_state)
        if self.buffer is not None:
            self.buffer.append(exp)

        if done:
            self.total_reward = self.cur_reward
            self.total_step = self.cur_step
            self.reset()
        else:
            self.state = next_state

        return exp

    def __iter__ (self):
        while True:
            exp = self.play_steps()
            yield exp

    def reward_step (self):
        res = (self.total_reward, self.total_step)
        self.total_reward = None
        self.total_step = None

        return res


class MultiExpSource:
    def __init__ (self, envs: List[gym.Env], agent: BaseAgent, buffer: ReplayBuffer=None, steps_count: int=2, gamma: float=0.99):
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        self.exp_sources = []
        for env in envs:
            exp_source = ExperienceSource(env, agent, buffer, steps_count, gamma)
            self.exp_sources.append(exp_source)

        self.total_rewards = []
        self.total_steps = []

        self.reset()

    def reset (self):
        for exp_source in self.exp_sources:
            exp_source.reset()

    def play_steps (self):
        for exp_source in self.exp_sources:
            exp = exp_source.play_steps()
            yield exp

    def __iter__ (self):
        while True:
            for exp_source in self.exp_sources:
                exp = exp_source.play_steps()
                yield exp

    def reward_step (self):
        for exp_source in self.exp_sources:
            reward, step = exp_source.reward_step()
            self.total_rewards.append(reward)
            self.total_steps.append(step)

        res = (self.total_rewards.copy(), self.total_steps.copy)
        self.total_rewards.clear()
        self.total_steps.clear()

        return res
