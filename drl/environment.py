from gym.envs.registration import spec
from mlagents_envs.base_env import ActionSpec, DecisionStep
from mlagents_envs.communicator_objects import agent_action_pb2
import numpy as np
import gym

import mlagents
from mlagents_envs.environment import UnityEnvironment, ActionTuple
from numpy.core.arrayprint import dtype_is_implied
from pygame.constants import HAT_RIGHT

class UnityEnv (gym.Env):
    def __init__(self, env: UnityEnvironment) -> None:
        super().__init__()
        self.env = env
        self.env.reset()
        self.behavior_name = list(self.env.behavior_specs)[0]
        self.spec = self.env.behavior_specs[self.behavior_name]

        self.obs_spec = self.spec.observation_specs
        self.act_spec = self.spec.action_spec

        obs_shape = self.obs_spec[0].shape
        self.observation_space = gym.spaces.Box(
            low = np.full(obs_shape, -np.inf),
            high = np.full(obs_shape, np.inf),
            dtype = np.float32
            )

        if self.act_spec.is_continuous():
            self.action_space = gym.spaces.Box(
                low = np.full(self.act_spec.continuous_size, -np.inf),
                high = np.full(self.act_spec.continuous_size, np.inf),
                dtype = np.float32
            )
        elif self.act_spec.is_discrete():
            self.action_space = gym.spaces.Discrete(self.act_spec.discrete_size)

        self.reset()

    def reset (self):
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        return decision_steps, terminal_steps

    def step (self, actions: np.ndarray):
        # actions = actions.reshape(1, -1)
        u_actions = ActionTuple(continuous=actions)
        # u_actions.add_continuous(actions)

        self.env.set_actions(self.behavior_name, u_actions)
        self.env.step()
        decision_steps, terminal_steps = self.env.get_steps(self.behavior_name)
        
        return decision_steps, terminal_steps

    def close (self):
        self.env.close()


