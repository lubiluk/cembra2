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
# from gym_recording.wrappers import TraceRecordingWrappe

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

class Extractor(nn.Module):
    def __init__(self):
        super(Extractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 27 * 37, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 27 * 37)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


env = TorchifyWrapper(
        TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
                  max_episode_steps=100))

ac_kwargs = dict(hidden_sizes=[64, 64, 64],
                 activation=nn.ReLU,
                 extractor_module=Extractor)
rb_kwargs = dict(size=50000)

logger_kwargs = dict(output_dir='data/2', exp_name='2')

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
            update_after=1,
            update_every=1,
            logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=1000, epochs=5000)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/2')

run_policy(env, get_action)