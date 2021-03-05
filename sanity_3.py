import gym
import gym_pepper
import torch as th
import torch.nn as nn
from stable_baselines3 import HER, SAC
from utils.wrappers import DoneOnSuccessWrapper
from gym.wrappers.time_limit import TimeLimit

env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
                max_episode_steps=100)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[64, 64],
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1000000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
<<<<<<< HEAD
    ent_coef='auto',
=======
    ent_coef=0.001,
    n_sampled_goal=4,
    goal_selection_strategy='future',
>>>>>>> sanity
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