import gym
import numpy as np
from gym import spaces


class PepperPreprocessing(gym.ObservationWrapper):
    def __init__(self, env, depth_camera=False, top_camera=False):
        super(PepperPreprocessing, self).__init__(env)
        self._depth_camera = depth_camera
        self._top_camera = top_camera

        n_channels = 4

        if self._depth_camera:
            n_channels += 1

        if self._top_camera:
            n_channels += 3

        self.observation_space = spaces.Box(
            -np.inf, np.inf, shape=(n_channels, 240, 320), dtype=np.float32
        )

    def observation(self, observation):
        cam_bottom = observation['camera_bottom']
        joints_state = observation['joints_state']
        zeros = np.zeros(cam_bottom.shape[:2])
        zeros[:joints_state.shape[0],0] = joints_state

        cam_bot_nom = cam_bottom / 255 - 0.5
        stack = [cam_bot_nom, zeros]

        if self._depth_camera:
            cam_depth = observation['camera_depth']
            cam_dep_nom = cam_depth / 65535 - 0.5
            stack.insert(1, cam_depth)

        if self._top_camera:
            cam_top = observation['camera_top']
            cam_top_nom = cam_top / 255 - 0.5
            stack.insert(0, cam_top)

        stacked = np.dstack(stack)
        transposed = stacked.transpose(2,0,1)

        return transposed

