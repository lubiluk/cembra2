import gym
import gym_pepper
import torch
import torch.nn as nn
from algos import td3_her
from algos.td3_her import core
from gym.wrappers.time_limit import TimeLimit

<<<<<<< HEAD
=======
th.backends.cudnn.benchmark = True
th.autograd.set_detect_anomaly(False)
th.autograd.profiler.profile(enabled=False)

env = PepperPreprocessing(
    TimeLimit(gym.make("PepperReachCam-v0", gui=False), max_episode_steps=100)
)
>>>>>>> fd607bc (production settings)

def env_fn():
    return TimeLimit(gym.make("PepperPush-v0", gui=False),
                     max_episode_steps=100)


ac_kwargs = dict(hidden_sizes=[64, 64, 64], activation=nn.ReLU)

logger_kwargs = dict(output_dir='data/1', exp_name='1')

<<<<<<< HEAD
td3_her(env_fn=env_fn,
        actor_critic=core.MLPActorCritic,
        ac_kwargs=ac_kwargs,
        steps_per_epoch=10000,
        max_ep_len=100,
        epochs=100,
        batch_size=64,
        replay_size=100000,
        gamma=0.95,
        pi_lr=1e-3,
        q_lr=1e-3,
        update_after=1000,
        update_every=64,
        num_additional_goals=4,
        goal_selection_strategy='future',
        logger_kwargs=logger_kwargs)
=======
        # Compute shape by doing one forward pass
        with th.no_grad():
            obs = th.as_tensor(observation_space.sample()[None]).float()
            bottom_cam_obs = self._get_bottom_camera_obs(obs)
            bottom_n_flatten = self.bottom_cnn(bottom_cam_obs).shape[1]

        self.bottom_linear = nn.Sequential(
            nn.Linear(bottom_n_flatten, features_dim), nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        bottom = self.bottom_linear(
            self.bottom_cnn(
                self._get_bottom_camera_obs(observations)
            )
        )
        joints = self._get_joints_state_obs(observations)

        cat = th.cat((bottom, joints), dim=1)

        return cat

    def _get_bottom_camera_obs(self, obs):
        return obs[:,:3,:,:]

    def _get_joints_state_obs(self, obs):
        return obs[:,3,:18,0]


policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
    activation_fn=th.nn.ReLU,
    net_arch=[256, 256, 256],
)

model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    buffer_size=1000,
    batch_size=256,
    learning_rate=0.001,
    learning_starts=1000,
    gamma=0.95,
    ent_coef="auto",
    policy_kwargs=policy_kwargs,
    train_freq=1,
    tensorboard_log="./data/1_tensorboard/",
)


for _ in range(1000):
    model.learn(1000)
    model.save("./data/1")

>>>>>>> fd607bc (production settings)
