import gym
import gym_pepper
import torch
import torch.nn as nn
import torch.nn.functional as F
from algos import SAC
from algos.sac import core_cam
from algos.common import replay_buffer_cam
from gym.wrappers.time_limit import TimeLimit
from utils.wrappers import TorchifyWrapper

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

env = TorchifyWrapper(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
              max_episode_steps=100))


class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()

        obs_space = env.observation_space.spaces["camera"]

        self.cnn = nn.Sequential(
            nn.Conv2d(obs_space.shape[0], 8, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            obs = torch.as_tensor(
                obs_space.sample()
                [None]).float()
            n_flatten = self.cnn(obs).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, 64), nn.ReLU())

    def forward(self, x):
        x = self.linear(self.cnn(x))
        return x


ac_kwargs = dict(hidden_sizes=[256, 256],
                 activation=nn.ReLU,
                 extractor_module=Extractor)
rb_kwargs = dict(size=40000)

logger_kwargs = dict(output_dir='data/reach_cam', exp_name='reach_cam')

model = SAC(env=env,
            actor_critic=core_cam.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer_cam.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=100,
            batch_size=256,
            gamma=0.95,
            lr=0.0003,
            update_after=512,
            update_every=512,
            logger_kwargs=logger_kwargs,
            use_gpu_buffer=True)

model.train(steps_per_epoch=1024, epochs=5000)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/reach_cam')

run_policy(env, get_action)
