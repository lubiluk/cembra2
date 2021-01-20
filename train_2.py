import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import sac
from utils import wrappers
from utils.pepper_preprocessing import PepperPreprocessing
import algos.cnn as core
from gym.wrappers.time_limit import TimeLimit


def env_fn():
    return PepperPreprocessing(
        TimeLimit(gym.make("PepperReachCam-v0", gui=False),
                  max_episode_steps=100))


e_kwargs = dict(cnn_sizes=[], activation=nn.ReLU)
ac_kwargs = dict(hidden_sizes=[64, 64],
                 activation=nn.ReLU,
                 extractor=core.FeatureExtractor,
                 e_kwargs=e_kwargs)

logger_kwargs = dict(output_dir='data/2', exp_name='2')

sac(env_fn=env_fn,
    actor_critic=core.MLPActorCritic,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=1000,
    max_ep_len=100,
    epochs=10,
    batch_size=64,
    replay_size=1000,
    gamma=0.95,
    lr=0.001,
    update_after=100,
    update_every=1,
    logger_kwargs=logger_kwargs)
