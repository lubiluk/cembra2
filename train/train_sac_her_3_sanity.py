from algos import sac_her
import torch as th
import gym



# For debugging only
# envs = iter([
#     gym.make('PepperPush-v0', sim_steps_per_action=10, gui=True),
#     gym.make('PepperPush-v0', sim_steps_per_action=10)
# ])


# def env_fn(): 
#     return next(envs)

def env_fn():
    return gym.make('FetchPush-v1')

ac_kwargs = dict(hidden_sizes=[64, 64], activation=th.nn.ReLU)

logger_kwargs = dict(
    output_dir='data/fetch_push_sac_her_3',
    exp_name='fetch_push_sac_her_3')

sac_her(
    env_fn=env_fn,
    ac_kwargs=ac_kwargs,
    steps_per_epoch=15000,
    max_ep_len=300,
    epochs=200,
    batch_size=256,
    replay_size=1000000,
    gamma=0.95,
    lr=0.001,
    update_after=1000,
    update_every=1,
    num_additional_goals=4,
    goal_selection_strategy='future',
    logger_kwargs=logger_kwargs)
