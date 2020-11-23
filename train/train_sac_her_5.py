from stable_baselines3 import HER, SAC
import gym
import gym_pepper
import torch as th

env = gym.make('PepperPush-v0')

model = HER('MlpPolicy',
            env,
            SAC,
            online_sampling=False,
            verbose=1,
            max_episode_length=300,
            buffer_size=1000000,
            batch_size=256,
            learning_rate=0.001,
            learning_starts=1000,
            n_sampled_goal=4,
            gamma=0.95,
            goal_selection_strategy='future',
            ent_coef='auto',
            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[128, 128]),
            train_freq=1,
            tensorboard_log="./data/pepper_push_sac_her_5_tensorboard/"
            )
# Train the model
model.learn(3000000)

model.save("./data/pepper_push_sac_her_5")

