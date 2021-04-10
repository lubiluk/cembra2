import os

import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from utils.wrappers import DoneOnSuccessWrapper

save_path = "./trained/reach_sb"

env = TimeLimit(gym.make("PepperReach-v0", gui=True, dense=True, head_motion=False),
                      max_episode_steps=100)
model = SAC.load(save_path)
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
