import torch
import numpy as np
from .utils import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_space, act_dim, env=None, size=100000, device=None):
        self.device = device

        img_dim = obs_space.spaces["camera_bottom"].shape
        lin_dim = obs_space.spaces["joints_state"].shape

        self.obs_img_buf = torch.zeros(combined_shape(size, img_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs_lin_buf = torch.zeros(combined_shape(size, lin_dim),
                                       dtype=torch.float32,
                                       device=device)
        self.obs2_img_buf = torch.zeros(combined_shape(size, img_dim),
                                        dtype=torch.float32,
                                        device=device)
        self.obs2_lin_buf = torch.zeros(combined_shape(size, lin_dim),
                                        dtype=torch.float32,
                                        device=device)
        self.act_buf = torch.zeros(combined_shape(size, act_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, info):
        obs_img = obs["camera_bottom"]
        obs_lin = obs["joints_state"]

        next_obs_img = next_obs["camera_bottom"]
        next_obs_lin = next_obs["joints_state"]

        self.obs_img_buf[self.ptr] = torch.as_tensor(obs_img,
                                                     dtype=torch.float32)
        self.obs_lin_buf[self.ptr] = torch.as_tensor(obs_lin,
                                                     dtype=torch.float32)
        self.obs2_img_buf[self.ptr] = torch.as_tensor(next_obs_img,
                                                      dtype=torch.float32)
        self.obs2_lin_buf[self.ptr] = torch.as_tensor(next_obs_lin,
                                                      dtype=torch.float32)
        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        obs = {
            "camera_bottom": self.obs_img_buf[idxs],
            "joints_state": self.obs_lin_buf[idxs]
        }
        obs2 = {
            "camera_bottom": self.obs2_img_buf[idxs],
            "joints_state": self.obs2_lin_buf[idxs]
        }

        return dict(obs=obs,
                    obs2=obs2,
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def start_episode(self):
        pass

    def end_episode(self):
        pass