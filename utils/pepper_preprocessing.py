import gym
import numpy as np
from gym import spaces


class PepperPreprocessing(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(8, 240, 320), dtype=np.float32
        )

    def observation(self, observation):
        cam_top = observation['camera_top']
        cam_bottom = observation['camera_bottom']
        cam_depth = observation['camera_depth']
        joints_state = observation['joints_state']
        zeros = np.zeros(cam_depth.shape)
        zeros[:joints_state.shape[0],0] = joints_state

        stacked = np.dstack([cam_top, cam_bottom, cam_depth, zeros])
        transposed = stacked.transpose(2,0,1)

        return transposed
        
