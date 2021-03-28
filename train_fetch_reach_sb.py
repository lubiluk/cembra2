import gym
import numpy as np
import torch as th
import torch.nn as nn

from stable_baselines3 import HER, SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make('FetchReach-v1')

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[64, 64],
)

model = HER(
    MlpPolicy,
    env,
    SAC,
    online_sampling=False,
    verbose=1,
    max_episode_length=100,
    buffer_size=1000000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    n_sampled_goal=4,
    goal_selection_strategy='future',
    policy_kwargs=policy_kwargs
)

model.learn(total_timesteps=20000)
model.save("data/fetch_reach_sb")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
