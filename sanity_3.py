import gym
import gym_pepper
import torch as th
import torch.nn as nn
from stable_baselines3 import HER, SAC
from utils.wrappers import DoneOnSuccessWrapper

env = DoneOnSuccessWrapper(gym.make("FetchPush-v1"))

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[64, 64],
)

model = HER(
    "MlpPolicy",
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
    ent_coef=0.2,
    n_sampled_goal=4,
    goal_selection_strategy='future',
    policy_kwargs=policy_kwargs,
    train_freq=1,
)

model.learn(1000000)

model.save("./data/sanity_3")

model = HER.load("./data/sanity_3")

obs = env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()