from stable_baselines3 import SAC
import gym
import gym_pepper
from gym.wrappers.time_limit import TimeLimit
from utils.pepper_preprocessing import PepperPreprocessing

test_env = PepperPreprocessing(
    TimeLimit(gym.make("PepperReachCam-v0", gui=True), max_episode_steps=100)
)
model = SAC.load("./data/1", env=test_env)

obs = test_env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)

    if done:
        obs = test_env.reset()