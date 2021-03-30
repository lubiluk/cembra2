import os

import gym
import gym_pepper
import matplotlib.pyplot as plt
import torch as th
import torch.nn as nn
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import HER, SAC
from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import plot_results

from utils.callbacks import SaveOnBestTrainingRewardCallback
from utils.extractors import CustomCNN
from utils.wrappers import DepthWrapper

log_dir = "./data/reach_depth_sb"

os.makedirs(log_dir, exist_ok=True)

env = DepthWrapper(
    TimeLimit(gym.make("PepperReachDepth-v0", gui=True, dense=True),
              max_episode_steps=100))
env = Monitor(env, log_dir)

policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[64, 64],
                     features_extractor_class=CustomCNN,
                     features_extractor_kwargs=dict(features_dim=16,
                                                    linear_dim=16,
                                                    n_channels=1),
                     normalize_images=False)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=100000,
    batch_size=256,
    learning_rate=0.0003,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    policy_kwargs=policy_kwargs,
    train_freq=1,
)

callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_dir)

timesteps = 1000_000

model.learn(timesteps)

model.save(log_dir)

plot_results([log_dir], timesteps, results_plotter.X_TIMESTEPS,
             "SAC PepperReach")
plt.show()

env = TimeLimit(gym.make("PepperReachDepth-v0", gui=True, dense=True),
                max_episode_steps=100)
model = SAC.load(log_dir + ".zip")
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
