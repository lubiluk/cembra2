import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import td3_cam
from algos.td3_cam import core
from gym.wrappers.time_limit import TimeLimit
from utils.wrappers import GrayscaleWrapper, AntiTorchWrapper
from utils.framebuffer import FrameBuffer


def env_fn():
    return FrameBuffer(AntiTorchWrapper(
        GrayscaleWrapper(
            TimeLimit(gym.make("PepperReachCam-v0", gui=False, dense=True),
                      max_episode_steps=100))),
                       n_frames=4)


ac_kwargs = dict(hidden_sizes=[128, 128],
                 activation=nn.ReLU,
                 conv_sizes=[[4, 32, 8, 4, 0], [32, 64, 4, 2, 0]])

logger_kwargs = dict(output_dir='data/2', exp_name='2')

td3_cam(env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=10000,
        max_ep_len=100,
        epochs=100,
        batch_size=64,
        replay_size=20000,
        gamma=0.95,
        pi_lr=1e-3,
        q_lr=1e-3,
        update_after=1000,
        update_every=64,
        logger_kwargs=logger_kwargs)
