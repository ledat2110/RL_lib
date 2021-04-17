import gym
import torch
import random
import collections
from torch.autograd import Variable

import numpy as np

from collections import namedtuple, deque

from .agent import BaseAgent
from .common import utils
from . import agent

# one single experience step
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'done'])
EpisodeEnded = namedtuple("EpisodeEnded", field_names=['reward', 'step', 'epsilon'])


class ExperienceSource:
    """
    Simple n-step experience source using single or multiple environments

    Every experience contains n list of Experience entries
    """
    def __init__(self, env, agent, steps_count=2, steps_delta=1, vectorized=False):
        """
        Create simple experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions to take
        :param steps_count: count of steps to track for every experience chain
        :param steps_delta: how many steps to do between experience items
        :param vectorized: support of vectorized envs from OpenAI universe
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(steps_count, int)
        assert steps_count >= 1
        assert isinstance(vectorized, bool)
        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.steps_count = steps_count
        self.steps_delta = steps_delta
        self.total_rewards = []
        self.total_steps = []
        self.vectorized = vectorized

    def __iter__(self):
        states, agent_states, histories, cur_rewards, cur_steps = [], [], [], [], []
        env_lens = []
        for env in self.pool:
            obs = env.reset()
            # if the environment is vectorized, all it's output is lists of results.
            # Details are here: https://github.com/openai/universe/blob/master/doc/env_semantics.rst
            if self.vectorized:
                obs_len = len(obs)
                states.extend(obs)
            else:
                obs_len = 1
                states.append(obs)
            env_lens.append(obs_len)

            for _ in range(obs_len):
                histories.append(deque(maxlen=self.steps_count))
                cur_rewards.append(0.0)
                cur_steps.append(0)
                agent_states.append(self.agent.initial_state())

        iter_idx = 0
        while True:
            actions = [None] * len(states)
            states_input = []
            states_indices = []
            for idx, state in enumerate(states):
                if state is None:
                    actions[idx] = self.pool[0].action_space.sample()  # assume that all envs are from the same family
                else:
                    states_input.append(state)
                    states_indices.append(idx)
            if states_input:
                states_actions, new_agent_states = self.agent(states_input, agent_states)
                for idx, action in enumerate(states_actions):
                    g_idx = states_indices[idx]
                    actions[g_idx] = action
                    agent_states[g_idx] = new_agent_states[idx]
            grouped_actions = _group_list(actions, env_lens)

            global_ofs = 0
            for env_idx, (env, action_n) in enumerate(zip(self.pool, grouped_actions)):
                if self.vectorized:
                    next_state_n, r_n, is_done_n, _ = env.step(action_n)
                else:
                    next_state, r, is_done, _ = env.step(action_n[0])
                    next_state_n, r_n, is_done_n = [next_state], [r], [is_done]

                for ofs, (action, next_state, r, is_done) in enumerate(zip(action_n, next_state_n, r_n, is_done_n)):
                    idx = global_ofs + ofs
                    state = states[idx]
                    history = histories[idx]

                    cur_rewards[idx] += r
                    cur_steps[idx] += 1
                    if state is not None:
                        history.append(Experience(state=state, action=action, reward=r, done=is_done))
                    if len(history) == self.steps_count and iter_idx % self.steps_delta == 0:
                        yield tuple(history)
                    states[idx] = next_state
                    if is_done:
                        # in case of very short episode (shorter than our steps count), send gathered history
                        if 0 < len(history) < self.steps_count:
                            yield tuple(history)
                        # generate tail of history
                        while len(history) > 1:
                            history.popleft()
                            yield tuple(history)
                        self.total_rewards.append(cur_rewards[idx])
                        self.total_steps.append(cur_steps[idx])
                        cur_rewards[idx] = 0.0
                        cur_steps[idx] = 0
                        # vectorized envs are reset automatically
                        states[idx] = env.reset() if not self.vectorized else None
                        agent_states[idx] = self.agent.initial_state()
                        history.clear()
                global_ofs += len(action_n)
            iter_idx += 1

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


def _group_list(items, lens):
    """
    Unflat the list of items by lens
    :param items: list of items
    :param lens: list of integers
    :return: list of list of items grouped by lengths
    """
    res = []
    cur_ofs = 0
    for g_len in lens:
        res.append(items[cur_ofs:cur_ofs+g_len])
        cur_ofs += g_len
    return res


# those entries are emitted from ExperienceSourceFirstLast. Reward is discounted over the trajectory piece
ExperienceFirstLast = collections.namedtuple('ExperienceFirstLast', ('state', 'action', 'reward', 'last_state'))


class ExperienceSourceFirstLast(ExperienceSource):
    """
    This is a wrapper around ExperienceSource to prevent storing full trajectory in replay buffer when we need
    only first and last states. For every trajectory piece it calculates discounted reward and emits only first
    and last states and action taken in the first state.

    If we have partial trajectory at the end of episode, last_state will be None
    """
    def __init__(self, env, agent, gamma, steps_count=1, steps_delta=1, vectorized=False):
        assert isinstance(gamma, float)
        super(ExperienceSourceFirstLast, self).__init__(env, agent, steps_count+1, steps_delta, vectorized=vectorized)
        self.gamma = gamma
        self.steps = steps_count

    def __iter__(self):
        for exp in super(ExperienceSourceFirstLast, self).__iter__():
            if exp[-1].done and len(exp) <= self.steps:
                last_state = None
                elems = exp
            else:
                last_state = exp[-1].state
                elems = exp[:-1]
            total_reward = 0.0
            for e in reversed(elems):
                total_reward *= self.gamma
                total_reward += e.reward
            yield ExperienceFirstLast(state=exp[0].state, action=exp[0].action,
                                      reward=total_reward, last_state=last_state)


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


class ExperienceSourceRollouts:
    """
    N-step rollout experience source following A3C rollouts scheme. Have to be used with agent,
    keeping the value in its state (for example, agent.ActorCriticAgent).

    Yields batches of num_envs * n_steps samples with the following arrays:
    1. observations
    2. actions
    3. discounted rewards, with values approximation
    4. values
    """
    def __init__(self, env, agent, gamma, steps_count=5):
        """
        Constructs the rollout experience source
        :param env: environment or list of environments to be used
        :param agent: callable to convert batch of states into actions
        :param steps_count: how many steps to perform rollouts
        """
        assert isinstance(env, (gym.Env, list, tuple))
        assert isinstance(agent, BaseAgent)
        assert isinstance(gamma, float)
        assert isinstance(steps_count, int)
        assert steps_count >= 1

        if isinstance(env, (list, tuple)):
            self.pool = env
        else:
            self.pool = [env]
        self.agent = agent
        self.gamma = gamma
        self.steps_count = steps_count
        self.total_rewards = []
        self.total_steps = []

    def __iter__(self):
        pool_size = len(self.pool)
        states = [np.array(e.reset()) for e in self.pool]
        mb_states = np.zeros((pool_size, self.steps_count) + states[0].shape, dtype=states[0].dtype)
        mb_rewards = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_values = np.zeros((pool_size, self.steps_count), dtype=np.float32)
        mb_actions = np.zeros((pool_size, self.steps_count), dtype=np.int64)
        mb_dones = np.zeros((pool_size, self.steps_count), dtype=np.bool)
        total_rewards = [0.0] * pool_size
        total_steps = [0] * pool_size
        agent_states = None
        step_idx = 0

        while True:
            actions, agent_states = self.agent(states, agent_states)
            rewards = []
            dones = []
            new_states = []
            for env_idx, (e, action) in enumerate(zip(self.pool, actions)):
                o, r, done, _ = e.step(action)
                total_rewards[env_idx] += r
                total_steps[env_idx] += 1
                if done:
                    o = e.reset()
                    self.total_rewards.append(total_rewards[env_idx])
                    self.total_steps.append(total_steps[env_idx])
                    total_rewards[env_idx] = 0.0
                    total_steps[env_idx] = 0
                new_states.append(np.array(o))
                dones.append(done)
                rewards.append(r)
            # we need an extra step to get values approximation for rollouts
            if step_idx == self.steps_count:
                # calculate rollout rewards
                for env_idx, (env_rewards, env_dones, last_value) in enumerate(zip(mb_rewards, mb_dones, agent_states)):
                    env_rewards = env_rewards.tolist()
                    env_dones = env_dones.tolist()
                    if not env_dones[-1]:
                        env_rewards = discount_with_dones(env_rewards + [last_value], env_dones + [False], self.gamma)[:-1]
                    else:
                        env_rewards = discount_with_dones(env_rewards, env_dones, self.gamma)
                    mb_rewards[env_idx] = env_rewards
                yield mb_states.reshape((-1,) + mb_states.shape[2:]), mb_rewards.flatten(), mb_actions.flatten(), mb_values.flatten()
                step_idx = 0
            mb_states[:, step_idx] = states
            mb_rewards[:, step_idx] = rewards
            mb_values[:, step_idx] = agent_states
            mb_actions[:, step_idx] = actions
            mb_dones[:, step_idx] = dones
            step_idx += 1
            states = new_states

    def pop_total_rewards(self):
        r = self.total_rewards
        if r:
            self.total_rewards = []
            self.total_steps = []
        return r

    def pop_rewards_steps(self):
        res = list(zip(self.total_rewards, self.total_steps))
        if res:
            self.total_rewards, self.total_steps = [], []
        return res


class ExperienceSourceBuffer:
    """
    The same as ExperienceSource, but takes episodes from the buffer
    """
    def __init__(self, buffer, steps_count=1):
        """
        Create buffered experience source
        :param buffer: list of episodes, each is a list of Experience object
        :param steps_count: count of steps in every entry
        """
        self.update_buffer(buffer)
        self.steps_count = steps_count

    def update_buffer(self, buffer):
        self.buffer = buffer
        self.lens = list(map(len, buffer))

    def __iter__(self):
        """
        Infinitely sample episode from the buffer and then sample item offset
        """
        while True:
            episode = random.randrange(len(self.buffer))
            ofs = random.randrange(self.lens[episode] - self.steps_count - 1)
            yield self.buffer[episode][ofs:ofs+self.steps_count]


class ExperienceReplayBuffer:
    def __init__(self, experience_source, buffer_size):
        assert isinstance(experience_source, (ExperienceSource, type(None)))
        assert isinstance(buffer_size, int)
        self.experience_source_iter = None if experience_source is None else iter(experience_source)
        self.buffer = []
        self.capacity = buffer_size
        self.pos = 0

    def __len__(self):
        return len(self.buffer)

    def __iter__(self):
        return iter(self.buffer)

    def sample(self, batch_size):
        """
        Get one random batch from experience replay
        TODO: implement sampling order policy
        :param batch_size:
        :return:
        """
        if len(self.buffer) <= batch_size:
            return self.buffer
        # Warning: replace=False makes random.choice O(n)
        keys = np.random.choice(len(self.buffer), batch_size, replace=True)
        return [self.buffer[key] for key in keys]

    def _add(self, sample):
        if len(self.buffer) < self.capacity:
            self.buffer.append(sample)
        else:
            self.buffer[self.pos] = sample
        self.pos = (self.pos + 1) % self.capacity

    def populate(self, samples):
        """
        Populates samples into the buffer
        :param samples: how many samples to populate
        """
        for _ in range(samples):
            entry = next(self.experience_source_iter)
            self._add(entry)

class PrioReplayBufferNaive:
    def __init__(self, exp_source, buf_size, prob_alpha=0.6):
        self.exp_source_iter = iter(exp_source)
        self.prob_alpha = prob_alpha
        self.capacity = buf_size
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((buf_size, ), dtype=np.float32)

    def __len__(self):
        return len(self.buffer)

    def populate(self, count):
        max_prio = self.priorities.max() if self.buffer else 1.0
        for _ in range(count):
            sample = next(self.exp_source_iter)
            if len(self.buffer) < self.capacity:
                self.buffer.append(sample)
            else:
                self.buffer[self.pos] = sample
            self.priorities[self.pos] = max_prio
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        probs = np.array(prios, dtype=np.float32) ** self.prob_alpha

        probs /= probs.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs, replace=True)
        samples = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, np.array(weights, dtype=np.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio


class PrioritizedReplayBuffer(ExperienceReplayBuffer):
    def __init__(self, experience_source, buffer_size, alpha):
        super(PrioritizedReplayBuffer, self).__init__(experience_source, buffer_size)
        assert alpha > 0
        self._alpha = alpha

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = utils.SumSegmentTree(it_capacity)
        self._it_min = utils.MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def _add(self, *args, **kwargs):
        idx = self.pos
        super()._add(*args, **kwargs)
        self._it_sum[idx] = self._max_priority ** self._alpha
        self._it_min[idx] = self._max_priority ** self._alpha

    def _sample_proportional(self, batch_size):
        res = []
        for _ in range(batch_size):
            mass = random.random() * self._it_sum.sum(0, len(self) - 1)
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self, batch_size, beta):
        assert beta > 0

        idxes = self._sample_proportional(batch_size)

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self)) ** (-beta)
            weights.append(weight / max_weight)
        weights = np.array(weights, dtype=np.float32)
        samples = [self.buffer[idx] for idx in idxes]
        return samples, idxes, weights

    def update_priorities(self, idxes, priorities):
        assert len(idxes) == len(priorities)
        for idx, priority in zip(idxes, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)
            self._it_sum[idx] = priority ** self._alpha
            self._it_min[idx] = priority ** self._alpha

            self._max_priority = max(self._max_priority, priority)

def unpack_batch_a2c(batch, net, last_val_gamma, device="cpu"):
    """
    Convert batch into training tensors
    :param batch:
    :param net:
    :return: states variable, actions tensor, reference values variable
    """
    states = []
    actions = []
    rewards = []
    not_done_idx = []
    last_states = []
    for idx, exp in enumerate(batch):
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        if exp.last_state is not None:
            not_done_idx.append(idx)
            last_states.append(exp.last_state)
    states_v = agent.float32_preprocessor(states).to(device)
    actions_v = torch.FloatTensor(actions).to(device)

    # handle rewards
    rewards_np = np.array(rewards, dtype=np.float32)
    if not_done_idx:
        last_states_v = agent.float32_preprocessor(last_states).to(device)
        last_vals_v = net(last_states_v)
        last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
        rewards_np[not_done_idx] += last_val_gamma * last_vals_np

    ref_vals_v = torch.FloatTensor(rewards_np).to(device)
    return states_v, actions_v, ref_vals_v


def unpack_batch_dqn(batch, device="cpu"):
    states, actions, rewards, dones, last_states = [], [], [], [], []
    for exp in batch:
        states.append(exp.state)
        actions.append(exp.action)
        rewards.append(exp.reward)
        dones.append(exp.last_state is None)
        if exp.last_state is None:
            last_states.append(exp.state)
        else:
            last_states.append(exp.last_state)
    states_v = agent.float32_preprocessor(states).to(device)
    actions_v = agent.float32_preprocessor(actions).to(device)
    rewards_v = agent.float32_preprocessor(rewards).to(device)
    last_states_v = agent.float32_preprocessor(last_states).to(device)
    dones_t = torch.BoolTensor(dones).to(device)
    return states_v, actions_v, rewards_v, dones_t, last_states_v

# class BatchData:
#     def __init__ (self, max_size: int):
#         self.batch = collections.deque(maxlen=max_size)
#         self.max_size = max_size

#     def add (self, value):
#         assert isinstance(value, (List, ExperienceFirstLast))
#         if isinstance(value, ExperienceFirstLast):
#             self.batch.append(value)
#         else:
#             self.batch.extend(value)

#     def dqn_unpack (self, device='cpu'):
#         states, actions, rewards, dones, last_states = [], [], [], [], []
#         for exp in self.batch:
#             states.append(exp.state)
#             actions.append(exp.action)
#             rewards.append(exp.reward)
#             dones.append(exp.last_state is None)
#             if exp.last_state is None:
#                 last_states.append(exp.state)
#             else:
#                 last_states.append(exp.last_state)
#         states_v = agent.float32_preprocessor(states).to(device)
#         actions_v = agent.float32_preprocessor(actions).to(device)
#         rewards_v = agent.float32_preprocessor(rewards).to(device)
#         last_states_v = agent.float32_preprocessor(last_states).to(device)
#         dones_t = torch.BoolTensor(dones).to(device)
#         return states_v, actions_v, rewards_v, dones_t, last_states_v

#     def a2c_unpack (self, net, last_val_gamma, device="cpu"):
#         """
#         Convert batch into training tensors
#         :param batch:
#         :param net:
#         :return: states variable, actions tensor, reference values variable
#         """
#         states = []
#         actions = []
#         rewards = []
#         not_done_idx = []
#         last_states = []
#         for idx, exp in enumerate(self.batch):
#             states.append(exp.state)
#             actions.append(exp.action)
#             rewards.append(exp.reward)
#             if exp.last_state is not None:
#                 not_done_idx.append(idx)
#                 last_states.append(exp.last_state)
#         states_v = agent.float32_preprocessor(states).to(device)
#         actions_v = torch.FloatTensor(actions).to(device)

#         # handle rewards
#         rewards_np = np.array(rewards, dtype=np.float32)
#         if not_done_idx:
#             last_states_v = agent.float32_preprocessor(last_states).to(device)
#             last_vals_v = net(last_states_v)[2]
#             last_vals_np = last_vals_v.data.cpu().numpy()[:, 0]
#             rewards_np[not_done_idx] += last_val_gamma * last_vals_np

#         ref_vals_v = torch.FloatTensor(rewards_np).to(device)
#         return states_v, actions_v, ref_vals_v

#     def clear (self):
#         self.batch.clear()

#     def __len__ (self):
#         return len(self.batch)

# def unpack_data (exps: List[Experience]) -> Tuple:
#     states, actions, rewards, dones, next_states = [], [], [], [], []
#     for exp in exps:
#         states.append(np.array(exp.state))
#         actions.append(exp.action)
#         rewards.append(exp.reward)
#         dones.append(exp.done)
#         next_states.append(np.array(exp.last_state))

#     states = np.array(states, copy=False)
#     actions = np.array(actions)
#     rewards = np.array(rewards, dtype=np.float32)
#     dones = np.array(dones, dtype=np.uint8)
#     next_states = np.array(next_states, copy=False)
    
#     return states, actions, rewards, dones, next_states

# class ReplayBuffer:
#     def __init__ (self, capacity: int):
#         assert isinstance(capacity, int)
#         self.capacity = capacity
#         self.buffer = collections.deque(maxlen=capacity)
    
#     def __len__ (self):
#         return len(self.buffer)

#     def sample(self, batch_size: int):
#         if len(self.buffer) <= batch_size:
#             samples = self.buffer
#         else:
#             indices = np.random.choice(len(self.buffer), batch_size, replace=False)
#             samples = [self.buffer[idx] for idx in indices]

#         samples = unpack_data(samples)

#         return samples
    
#     def append (self, exp: Experience):
#         self.buffer.append(exp)

#     @property
#     def size (self):
#         return self.capacity

# class PrioReplayBuffer:
#     def __init__ (self, capacity: int, prob_alpha: float=0.6, beta: float=0.4):
#         self.priorities = np.zeros((capacity, ), dtype=np.float32)
#         self.buffer = []
#         self.pos = 0
#         self.prob_alpha = prob_alpha
#         self.capacity = capacity
#         self.beta = beta

#     def __len__ (self):
#         return len(self.buffer)

#     def append(self, exp: Experience):
#         max_prio = self.priorities.max() if self.buffer else 1.0
#         if len(self.buffer) < self.capacity:
#             self.buffer.append(exp)
#         else:
#             self.buffer[self.pos] = exp
#         self.priorities[self.pos] = max_prio
#         self.pos = (self.pos + 1) % self.capacity

#     def sample(self, batch_size: int):
#         total = len(self.buffer)
#         probs = self.priorities[:total] ** self.prob_alpha
#         probs /= probs.sum()

#         if total <= batch_size:
#             indices = np.array(range(total))
#         else:
#             indices = np.random.choice(total, batch_size, p=probs, replace=False)
#         samples = [self.buffer[idx] for idx in indices]

#         weights = (total * probs[indices]) ** (-self.beta)
#         weights /= weights.max()

#         samples = unpack_data(samples)

#         return samples, indices, np.array(weights, dtype=np.float32)

#     def update_priorities (self, batch_indices: np.ndarray, batch_priorities: np.ndarray):
#         for idx, prio in zip(batch_indices, batch_priorities):
#             self.priorities[idx] = prio


# class ExperienceSource:
#     """
#     Simple n-step experience source using single or multiple environments

#     Every experience contains n list of Experience entries
#     """
#     def __init__ (self, env: gym.Env, agent: BaseAgent, buffer: ReplayBuffer=None, steps_count: int=2, gamma: float=0.99):
#         """
#         Create simple experience source
#         :param env: environment to be used
#         :param agent: callable to convert batch of states into actions to take
#         :param steps_count: count of steps to track for every experience chain
#         :param steps_delta: how many steps to do between experience items
#         :param vectorized: support of vectorized envs from OpenAI universe
#         """
#         assert isinstance(agent, BaseAgent)
#         assert isinstance(steps_count, int)
#         assert steps_count >= 1

#         self.agent = agent
#         self.steps_count = steps_count
#         self.gamma = gamma

#         self.env = env
#         self.buffer = buffer
#         self.history = collections.deque(maxlen=self.steps_count)

#         self.total_reward = None
#         self.total_step = None
#         self.reset()

#     def reset (self):
#         self.state = self.env.reset()
#         self.cur_step = 0
#         self.cur_reward = 0
#         self.history.clear()

#     def _create_exp (self):
#         state, action, done, last_state = self.history[0].state, self.history[0].action, self.history[-1].done, self.history[-1].last_state
#         reward = 0
#         for exp in reversed(self.history):
#             reward *= self.gamma
#             reward += exp.reward

#         exp = Experience(state, action, reward, done, last_state)
#         if self.buffer is not None:
#             self.buffer.append(exp)

#         return exp

#     def __iter__ (self):
#         while True:
#             state = self.state
#             action = self.agent(state)
#             next_state, reward, done, _ = self.env.step(action)

#             self.cur_reward += reward
#             self.cur_step += 1

#             exp = Experience(state, action, reward, done, next_state)
#             self.history.append(exp)
#             if len(self.history) == self.steps_count:
#                 exp = self._create_exp()
#                 yield exp
#             if done:
#                 while len(self.history) > 1:
#                     self.history.popleft()
#                     exp = self._create_exp()
#                     yield exp
#                 self.total_reward = self.cur_reward
#                 self.total_step = self.cur_step
#                 self.reset()
#             else:
#                 self.state = next_state

#     def reward_step (self):
#         res = (self.total_reward, self.total_step)
#         self.total_reward = None
#         self.total_step = None

#         return res


# class MultiExpSource:
#     def __init__ (self, envs: List[gym.Env], agent: BaseAgent, buffer: ReplayBuffer=None, steps_count: int=2, gamma: float=0.99):
#         assert isinstance(agent, BaseAgent)
#         assert isinstance(steps_count, int)
#         assert steps_count >= 1

#         self.exp_sources = []
#         self.iters = []
#         for env in envs:
#             exp_source = ExperienceSource(env, agent, buffer, steps_count, gamma)
#             self.exp_sources.append(exp_source)
#             self.iters.append(iter(exp_source))

#         self.total_rewards = []
#         self.total_steps = []

#         self.reset()

#     def reset (self):
#         for exp_source in self.exp_sources:
#             exp_source.reset()

#     def __iter__ (self):
#         idx = 0
#         while True:
#             exp = next(self.iters[idx])
#             if exp.done == False:
#                 idx = (idx + 1) % len(self.iters)
#             yield exp

#     def reward_step (self):
#         for exp_source in self.exp_sources:
#             reward, step = exp_source.reward_step()
#             if reward is not None:
#                 res = (reward, step)
#                 return res
#         return (None, None)
