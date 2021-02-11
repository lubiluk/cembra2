import gym
import numpy as np

from stable_baselines3 import SAC
from stable_baselines3.sac import MlpPolicy

env = gym.make('Pendulum-v0')

# model = SAC(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)
# model.save("data/sanity_0")

# del model  # remove to demonstrate saving and loading

model = SAC.load("data/sanity_0")

obs = env.reset()
while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        obs = env.reset()
