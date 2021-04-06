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

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 27 * 37, 64)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 27 * 37)
        x = F.relu(self.fc1(x))
        return x


env = TorchifyWrapper(
        TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
                  max_episode_steps=100))

ac_kwargs = dict(hidden_sizes=[64, 64, 64],
                 activation=nn.ReLU,
                 extractor_module=Extractor)
rb_kwargs = dict(size=1000000)

logger_kwargs = dict(output_dir='data/reach_cam', exp_name='reach_cam')

model = SAC(env=env,
            actor_critic=core_cam.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer_cam.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=100,
            batch_size=256,
            gamma=0.95,
            lr=0.001,
            alpha=0.0002,
            update_after=1024,
            update_every=512,
            logger_kwargs=logger_kwargs,
            use_gpu_buffer=False)

model.train(steps_per_epoch=1024, epochs=5000)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/reach_cam')

run_policy(env, get_action)