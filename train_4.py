from stable_baselines3 import TD3
import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit

env = TimeLimit(gym.make('PepperReach-v0', dense=True), max_episode_steps=100)

model = TD3('MlpPolicy',
            env,
            verbose=1,
            buffer_size=1000000,
            batch_size=256,
            learning_rate=0.001,
            learning_starts=1000,
            gamma=0.95,
            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64, 64]),
            tensorboard_log="./data/4_tensorboard/"
            )
# Train the model
model.learn(1000000)

model.save("./data/4")

