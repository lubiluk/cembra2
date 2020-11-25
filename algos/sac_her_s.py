from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import gym
import time
from . import core_sac as core
from .logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for DDPG agents.
    """

    def __init__(self, obs_dim, act_dim, goal_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(
            size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(
            size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.dgoal_buf = np.zeros(core.combined_shape(
            size, goal_dim), dtype=np.float32)
        self.agoal_buf = np.zeros(core.combined_shape(
            size, goal_dim), dtype=np.float32)
        self.info_buf = np.empty((size, 1), dtype=object)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done, dgoal, agoal, info):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.dgoal_buf[self.ptr] = dgoal
        self.agoal_buf[self.ptr] = agoal
        self.info_buf[self.ptr] = info
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    dgoal=self.dgoal_buf[idxs],
                    agoal=self.agoal_buf[idxs],
                    info=self.info_buf[idxs])

    def get_episode(self, ep_start_ptr, ep_len):
        idxs = np.arange(ep_start_ptr, min(ep_start_ptr + ep_len, self.size))

        if len(idxs) < ep_len:
            idxs = np.concatenate([idxs, np.arange((ep_start_ptr + ep_len) % self.size)])

        return dict(obs=self.obs_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    act=self.act_buf[idxs],
                    rew=self.rew_buf[idxs],
                    done=self.done_buf[idxs],
                    dgoal=self.dgoal_buf[idxs],
                    agoal=self.agoal_buf[idxs],
                    info=self.info_buf[idxs])




def sac_her_s(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1,
        num_additional_goals=1, goal_selection_strategy='final',
        demo_actions=[], demo_actions_repeat=0):
    """
    Soft Actor-Critic (SAC) with Hindsight Experience Repley (HER)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

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

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

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

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        num_additional_goals (int): Number of additional HER goals for replay.

        goal_selection_strategy (final, future, episode, random): Goal selection
            method for HER goal generation.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    
    obs_dim = env.observation_space.spaces["observation"].shape
    act_dim = env.action_space.shape[0]
    goal_dim = env.observation_space.spaces["desired_goal"].shape
    # The space of an observation concatenated with a goal
    low_val = np.concatenate([
        env.observation_space.spaces["observation"].low,
        env.observation_space.spaces["desired_goal"].low])
    high_val = np.concatenate([
        env.observation_space.spaces["observation"].high,
        env.observation_space.spaces["desired_goal"].high])
    og_space = gym.spaces.Box(low_val, high_val, dtype=np.float32)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(og_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, 
                                 goal_dim=goal_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o_dict, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            o = o_dict["observation"]
            g = o_dict["desired_goal"]
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                og = np.concatenate([o, g], axis=-1)
                o_dict, r, d, _ = test_env.step(get_action(og, True))
                o = o_dict["observation"]
                g = o_dict["desired_goal"]
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    def synthesize_experience(env, ep_start_ptr, ep_len, replay_buffer, num=1, selection_strategy='final'):
        ep = replay_buffer.get_episode(ep_start_ptr, ep_len)
        ep_end = len(ep['obs'])

        for idx in range(ep_end):
            obs = ep['obs'][idx]
            act = ep['act'][idx]
            obs2 = ep['obs2'][idx]
            agoal = ep['agoal'][idx]
            info = ep['info'][idx]

            for _ in range(num):
                if selection_strategy == 'final':
                    sel_idx = -1
                elif selection_strategy == 'future':
                    # We cannot sample a goal from the future in the last step of an episode
                    if idx == ep_end - 1:
                        break

                    dif_idxs = np.nonzero(ep['agoal'][idx+1:ep_end] != agoal)[0]

                    if dif_idxs.size == 0:
                        break

                    sel_idx = np.random.choice(dif_idxs)
                elif selection_strategy == 'episode':
                    dif_idxs = np.nonzero(ep['agoal'][:ep_end] != agoal)[0]

                    if dif_idxs.size == 0:
                        break

                    sel_idx = np.random.choice(dif_idxs)
                else:
                    raise ValueError(f"Unsupported selection_strategy: {selection_strategy}")

                sel_agoal = ep['agoal'][sel_idx]

                if not np.allclose(sel_agoal, agoal, rtol=0, atol=1e-4):
                    rew = env.compute_reward(agoal, sel_agoal, info)
                    replay_buffer.store(obs, act, rew, obs2, False, sel_agoal, agoal, info)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o_dict, ep_ret, ep_len = env.reset(), 0, 0
    ep_start_ptr = replay_buffer.ptr
    dd = False
    
    o = o_dict["observation"]
    dg = o_dict["desired_goal"]

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # First, use given demo actions
        if t < len(demo_actions * demo_actions_repeat):
            a = demo_actions[t % len(demo_actions)]
            dd = t % len(demo_actions) == 0 if t > 0 else False
        elif t > start_steps:
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards, 
            # use the learned policy. 
            odg = np.concatenate([o, dg], axis=-1)
            a = get_action(odg)
        else:
            a = env.action_space.sample()

        # Step the env
        o2_dict, r, d, i = env.step(a)
        ep_ret += r
        ep_len += 1
        o2 = o2_dict["observation"]
        dg2 = o2_dict["desired_goal"]
        ag2 = o2_dict["achieved_goal"]

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, dg, ag2, i)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2
        dg = dg2

        # End of trajectory handling
        if d or (ep_len == max_ep_len) or dd:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            synthesize_experience(env, ep_start_ptr=ep_start_ptr, ep_len=ep_len,
                                    replay_buffer=replay_buffer, num=num_additional_goals,
                                    selection_strategy=goal_selection_strategy)
            ep_start_ptr = replay_buffer.ptr
            o_dict, ep_ret, ep_len = env.reset(), 0, 0
            o = o_dict["observation"]
            dg = o_dict["desired_goal"]
            dd = False

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                og_batch = dict(
                    obs=torch.as_tensor(
                        np.concatenate([batch['obs'], batch['dgoal']], axis=-1), 
                        dtype=torch.float32),
                    obs2=torch.as_tensor(
                        np.concatenate([batch['obs2'], batch['dgoal']], axis=-1),
                        dtype=torch.float32),
                    act=torch.as_tensor(batch['act'], dtype=torch.float32),
                    rew=torch.as_tensor(batch['rew'], dtype=torch.float32),
                    done=torch.as_tensor(batch['done'], dtype=torch.float32)
                )
                update(data=og_batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

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
            logger.log_tabular('LogPi', with_min_and_max=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
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
    parser.add_argument('--exp_name', type=str, default='sac')
    args = parser.parse_args()

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    sac_her_s(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
