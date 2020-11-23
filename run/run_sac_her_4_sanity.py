from stable_baselines3 import HER
import gym
from utils import wrappers

test_env =  wrappers.DoneOnSuccessWrapper(gym.make('FetchPush-v1'))
model = HER.load('./trained/fetch_push_sac_her_4', env=test_env)

obs = test_env.reset()
for _ in range(1000):
    test_env.render()
    action, _ = model.model.predict(obs, deterministic=True)
    obs, reward, done, _ = test_env.step(action)

    if done:
        obs = test_env.reset()