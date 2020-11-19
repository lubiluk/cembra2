from stable_baselines3 import HER, SAC
import gym
import gym_pepper
import torch as th

env = gym.make('PepperPush-v0', sim_steps_per_action=10)

model = HER('MlpPolicy',
            env,
            SAC,
            online_sampling=True,
            verbose=1,
            max_episode_length=400,
            buffer_size=1000000,
            batch_size=256,
            learning_rate=0.001,
            learning_starts=1000,
            n_sampled_goal=4,
            gamma=0.95,
            goal_selection_strategy='future',
            ent_coef='auto',
            policy_kwargs = dict(activation_fn=th.nn.ReLU, net_arch=[256, 256, 256]),
            train_freq=1,
            tensorboard_log="./data/pepper_push_sac_her_1_tensorboard/"
            )
# Train the model
model.learn(3000000)

model.save("./data/pepper_push_sac_her_1")

