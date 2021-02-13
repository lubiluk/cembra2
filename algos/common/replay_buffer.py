import torch
import numpy as np
from .utils import combined_shape


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """
    def __init__(self, obs_space, act_dim, env=None, size=100000, device=None):
        self.device = device
        obs_dim = obs_space.shape
        self.obs_buf = torch.zeros(combined_shape(size, obs_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.obs2_buf = torch.zeros(combined_shape(size, obs_dim),
                                    dtype=torch.float32,
                                    device=device)
        self.act_buf = torch.zeros(combined_shape(size, act_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, info):
        self.obs_buf[self.ptr] = torch.as_tensor(obs, dtype=torch.float32)
        self.obs2_buf[self.ptr] = torch.as_tensor(next_obs,
                                                  dtype=torch.float32)
        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)
        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.float32)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs])

    def start_episode(self):
        pass

    def end_episode(self):
        pass