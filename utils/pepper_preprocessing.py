import gym
import numpy as np
from gym import spaces


class PepperPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(1, 3, 3), dtype=np.float32
        )

    def observation(self, observation):
        return self._stack_norm_observation(observation)

    def _stack_norm_observation(self, obs):
        return [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
