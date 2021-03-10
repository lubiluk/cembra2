import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import SAC
from algos.sac import core
from algos.common import replay_buffer
from gym.wrappers.time_limit import TimeLimit

torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(enabled=False)

env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
                max_episode_steps=100)

ac_kwargs = dict(hidden_sizes=[128, 128], activation=nn.ReLU)
rb_kwargs = dict(size=1000000)

logger_kwargs = dict(output_dir='data/0_2', exp_name='0_2')

model = SAC(env=env,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=ac_kwargs,
            replay_buffer=replay_buffer.ReplayBuffer,
            rb_kwargs=rb_kwargs,
            max_ep_len=100,
            batch_size=256,
            gamma=0.95,
            lr=0.001,
            alpha=0.0003,
            update_after=10,
            update_every=1,
            logger_kwargs=logger_kwargs)

model.train(steps_per_epoch=1000, epochs=1000)

from algos.test_policy import load_policy_and_env, run_policy

_, get_action = load_policy_and_env('data/0_2')

run_policy(env, get_action)