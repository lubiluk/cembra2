from stable_baselines3 import HER
import gym
import gym_pepper
from gym.wrappers.time_limit import TimeLimit

test_env = TimeLimit(gym.make('PepperReach-v0', dense=True, gui=True), max_episode_steps=100)
model = HER.load('./data/3', env=test_env)

obs = test_env.reset()
for _ in range(100):
    action, _ = model.model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)

    if done:
        obs = test_env.reset()