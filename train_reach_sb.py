import os

import gym
import gym_pepper
import torch as th
from gym.wrappers.time_limit import TimeLimit
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback

from utils.wrappers import DoneOnSuccessWrapper

log_dir = "./data/reach_sb_log"
save_path = "./data/reach_sb"
best_save_path = "./data/reach_sb_best"

os.makedirs(log_dir, exist_ok=True)

env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
                max_episode_steps=100)

eval_env = TimeLimit(gym.make("PepperReach-v0", gui=False, dense=True),
                max_episode_steps=100)

policy_kwargs = dict(
    activation_fn=th.nn.ReLU,
    net_arch=[256, 256, 256],
    normalize_images=False
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1_000_000,
    batch_size=256,
    learning_rate=0.0003,
    learning_starts=1000,
    gamma=0.95,
    ent_coef='auto',
    policy_kwargs=policy_kwargs,
    train_freq=1)

eval_callback = EvalCallback(eval_env,
                             best_model_save_path=best_save_path,
                             log_path=log_dir,
                             eval_freq=500,
                             deterministic=True,
                             render=False)

timesteps = 1_000_000

model.learn(timesteps, callback=eval_callback)

model.save(save_path)

# Evaluate
env.close()
env = TimeLimit(gym.make("PepperReach-v0", gui=True, dense=True),
                      max_episode_steps=100)
model = SAC.load(save_path)
obs = env.reset()
for _ in range(100):
    action, _states = model.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()
