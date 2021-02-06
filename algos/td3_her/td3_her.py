from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from . import core
from ..logx import EpochLogger
from signal import *
import sys


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 HER agents.
    """
    def __init__(self, obs_space, act_dim, size, device=None):
        self.device = device

        obs_dim = obs_space.spaces["observation"].shape[0]
        goal_dim = obs_space.spaces["desired_goal"].shape[0]

        self.obs_buf = torch.zeros(core.combined_shape(size, obs_dim),
                                   dtype=torch.float32,
                                   device=device)
        self.obs_dgoal_buf = torch.zeros(core.combined_shape(size, goal_dim),
                                         dtype=torch.float32,
                                         device=device)
        self.obs_agoal_buf = torch.zeros(core.combined_shape(size, goal_dim),
                                         dtype=torch.float32,
                                         device=device)

        self.act_buf = torch.zeros(core.combined_shape(size, act_dim),
                                   dtype=torch.float32,
                                   device=device)

        self.obs2_buf = torch.zeros(core.combined_shape(size, obs_dim),
                                    dtype=torch.float32,
                                    device=device)
        self.obs2_dgoal_buf = torch.zeros(core.combined_shape(size, goal_dim),
                                          dtype=torch.float32,
                                          device=device)
        self.obs2_agoal_buf = torch.zeros(core.combined_shape(size, goal_dim),
                                          dtype=torch.float32,
                                          device=device)

        self.rew_buf = torch.zeros(size, dtype=torch.float32, device=device)
        self.done_buf = torch.zeros(size, dtype=torch.bool, device=device)
        self.info_buf = np.empty((size, 1), dtype=object)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, info):
        obs_lin = obs["observation"]
        obs_dgoal = obs["desired_goal"]
        obs_agoal = obs["achieved_goal"]

        next_obs_lin = next_obs["observation"]
        next_obs_dgoal = next_obs["desired_goal"]
        next_obs_agoal = next_obs["achieved_goal"]

        self.obs_buf[self.ptr] = torch.as_tensor(obs_lin, dtype=torch.float32)
        self.obs_dgoal_buf[self.ptr] = torch.as_tensor(obs_dgoal,
                                                       dtype=torch.float32)
        self.obs_agoal_buf[self.ptr] = torch.as_tensor(obs_agoal,
                                                       dtype=torch.float32)

        self.obs2_buf[self.ptr] = torch.as_tensor(next_obs_lin,
                                                  dtype=torch.float32)
        self.obs2_dgoal_buf[self.ptr] = torch.as_tensor(next_obs_dgoal,
                                                        dtype=torch.float32)
        self.obs2_agoal_buf[self.ptr] = torch.as_tensor(next_obs_agoal,
                                                        dtype=torch.float32)

        self.act_buf[self.ptr] = torch.as_tensor(act, dtype=torch.float32)

        self.rew_buf[self.ptr] = torch.as_tensor(rew, dtype=torch.float32)
        self.done_buf[self.ptr] = torch.as_tensor(done, dtype=torch.bool)
        self.info_buf[self.ptr] = info

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)

        return self._get_batch(idxs)

    def get_episode(self, ep_start_ptr, ep_len):
        idxs = np.arange(ep_start_ptr, min(ep_start_ptr + ep_len, self.size))

        if len(idxs) < ep_len:
            idxs = np.concatenate(
                [idxs, np.arange((ep_start_ptr + ep_len) % self.size)])

        return self._get_batch(idxs)

    def _get_batch(self, idxs):
        obs = [{
            "observation": o,
            "desired_goal": dg,
            "achieved_goal": ag
        } for o, dg, ag in zip(self.obs_buf[idxs], self.obs_dgoal_buf[idxs],
                               self.obs_agoal_buf[idxs])]
        obs2 = [{
            "observation": o,
            "desired_goal": dg,
            "achieved_goal": ag
        } for o, dg, ag in zip(self.obs2_buf[idxs], self.obs2_dgoal_buf[idxs],
                               self.obs2_agoal_buf[idxs])]

        return dict(obs=obs,
                    obs2=obs2,
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    info=self.info_buf[idxs])


def td3_her(env_fn,
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(),
            seed=0,
            steps_per_epoch=4000,
            epochs=100,
            replay_size=int(1e6),
            gamma=0.99,
            polyak=0.995,
            pi_lr=1e-3,
            q_lr=1e-3,
            batch_size=100,
            start_steps=10000,
            update_after=1000,
            update_every=50,
            act_noise=0.1,
            target_noise=0.2,
            noise_clip=0.5,
            policy_delay=2,
            num_test_episodes=10,
            max_ep_len=1000,
            logger_kwargs=dict(),
            save_freq=1,
            num_additional_goals=1,
            goal_selection_strategy='final',
            num_updates=None,
            use_gpu_buffer=True,
            use_gpu_computation=True):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
        with Hindsight Experience Repley (HER)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        pi_lr (float): Learning rate for policy.

        q_lr (float): Learning rate for Q-networks.

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)

        target_noise (float): Stddev for smoothing noise added to target 
            policy.

        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.

        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        num_additional_goals (int): Number of additional HER goals for replay.

        goal_selection_strategy (final, future, episode, random): Goal selection
            method for HER goal generation.

        num_updates (int): The number of updates per `update_every`, 
            by default it is the same as save_freq

        use_gpu_buffer (bool): Whether or not to store replay buffer in GPU memory

        use_gpu_computation (bool): Whether or not to store computation graph on GPU memory

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    buff_device = torch.device("cpu")
    comp_device = torch.device("cpu")

    if torch.cuda.is_available():
        if use_gpu_buffer:
            buff_device = torch.device("cuda")
            logger.log('\nUsing GPU replay buffer\n')

        if use_gpu_computation:
            comp_device = torch.device("cuda")
            logger.log('\nUsing GPU computaion\n')
    else:
        logger.log('\nGPU unavailable\n')

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    test_env = env

    def clean(*args):
        env.close()
        sys.exit(0)

    for sig in (SIGABRT, SIGILL, SIGINT, SIGSEGV, SIGTERM):
        signal(sig, clean)

    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space,
                      env.action_space,
                      device=comp_device,
                      **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_space=env.observation_space,
                                 act_dim=act_dim,
                                 size=replay_size,
                                 device=buff_device)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(
        core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n' %
               var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data[
            'obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d.type(torch.int32)) * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().cpu().numpy(),
                         Q2Vals=q2.detach().cpu().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **loss_info)

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            logger.store(LossPi=loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(o)
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    def test_agent():
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                o, r, d, _ = test_env.step(get_action(o, 0))
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def synthesize_experience(env,
                              ep_start_ptr,
                              ep_len,
                              replay_buffer,
                              num=1,
                              selection_strategy='final'):
        ep = replay_buffer.get_episode(ep_start_ptr, ep_len)
        ep_end = len(ep['obs'])

        for idx in range(ep_end):
            obs = ep['obs'][idx]
            act = ep['act'][idx]
            obs2 = ep['obs2'][idx]
            info = ep['info'][idx]
            np_agoal = obs2['achieved_goal'].cpu().numpy()

            for _ in range(num):
                if selection_strategy == 'final':
                    sel_idx = -1
                elif selection_strategy == 'future':
                    # We cannot sample a goal from the future in the last step of an episode
                    if idx == ep_end - 1:
                        break
                    sel_idx = np.random.choice(np.arange(idx + 1, ep_end))
                elif selection_strategy == 'episode':
                    sel_idx = np.random.choice(np.arange(ep_end))
                else:
                    raise ValueError(
                        "Unsupported selection_strategy: {}".format(
                            selection_strategy))

                sel_obs2 = ep['obs2'][sel_idx]
                np_sel_agoal = sel_obs2['achieved_goal'].cpu().numpy()

                rew = env.compute_reward(np_agoal, np_sel_agoal, info)

                obs["desired_goal"] = sel_obs2["achieved_goal"]

                replay_buffer.store(obs, act, rew, obs2, False, info)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0
    ep_start_ptr = replay_buffer.ptr

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards,
        # use the learned policy (with some noise, via act_noise).
        if t > start_steps:
            a = get_action(o, act_noise)
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, i = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, i)

        # Super critical, easy to overlook step: make sure to update
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            synthesize_experience(env,
                                  ep_start_ptr=ep_start_ptr,
                                  ep_len=ep_len,
                                  replay_buffer=replay_buffer,
                                  num=num_additional_goals,
                                  selection_strategy=goal_selection_strategy)
            ep_start_ptr = replay_buffer.ptr
            o, ep_ret, ep_len = env.reset(), 0, 0

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(num_updates or update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t + 1) % steps_per_epoch == 0:
            epoch = (t + 1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            # Test the performance of the deterministic version of the agent.
            test_agent()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('TestEpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            logger.log_tabular('Q1Vals', with_min_and_max=True)
            logger.log_tabular('Q2Vals', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time() - start_time)
            logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='td3')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    td3_her(lambda: gym.make(args.env),
            actor_critic=core.MLPActorCritic,
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),
            gamma=args.gamma,
            seed=args.seed,
            epochs=args.epochs,
            logger_kwargs=logger_kwargs)
