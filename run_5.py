from stable_baselines3 import HER
import gym
import gym_pepper
from gym.wrappers.time_limit import TimeLimit

test_env = TimeLimit(gym.make('PepperPush-v0', gui=True), max_episode_steps=100)
model = HER.load('./data/5', env=test_env)

obs = test_env.reset()
for _ in range(1000):
    action, _ = model.model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)

    if done:
        obs = test_env.reset()