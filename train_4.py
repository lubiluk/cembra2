from stable_baselines3 import HER, SAC
import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit

env = TimeLimit(gym.make('PepperReach-v0', dense=False), max_episode_steps=100)

model = HER('MlpPolicy',
            env,
            SAC,
            online_sampling=False,
            verbose=1,
            max_episode_length=100,
            buffer_size=1000000,
            batch_size=256,
            learning_rate=0.001,
            learning_starts=1000,
            n_sampled_goal=4,
            gamma=0.95,
            goal_selection_strategy='future',
            ent_coef='auto',
            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[64, 64, 64]),
            train_freq=1,
            # tensorboard_log="./data/4_tensorboard/"
            )
# Train the model
model.learn(3000000)

model.save("./data/4")
