from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
import gym
import time
from .core import MLPActorCritic
from ..logx import EpochLogger
from ..common.replay_buffer import ReplayBuffer
from ..common.utils import count_vars
from algos.common import replay_buffer


class SAC:
    """ Soft Actor-Critic (SAC) """
    def __init__(self,
                 env,
                 actor_critic=MLPActorCritic,
                 replay_buffer=ReplayBuffer,
                 ac_kwargs=dict(),
                 rb_kwargs=dict(),
                 seed=0,
                 gamma=0.99,
                 polyak=0.995,
                 lr=1e-3,
                 ent_coef="auto",
                 target_entropy="auto",
                 batch_size=100,
                 start_steps=10000,
                 update_after=1000,
                 update_every=50,
                 num_test_episodes=10,
                 max_ep_len=1000,
                 logger_kwargs=dict(),
                 save_freq=1,
                 num_updates=None,
                 use_gpu_buffer=True,
                 use_gpu_computation=True):
        """
        Soft Actor-Critic (SAC)

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

            ent_coef (float): Entropy regularization coefficient. (Equivalent to 
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

            num_updates (int): The number of updates per `update_every`, 
                by default it is the same as save_freq

            use_gpu_buffer (bool): Whether or not to store replay buffer in GPU memory

            use_gpu_computation (bool): Whether or not to store computation graph on GPU memory

        """
        self.logger = EpochLogger(**logger_kwargs)

        best_logger_kwargs = logger_kwargs.copy()
        best_logger_kwargs["output_dir"] += '/best'
        self.best_logger = EpochLogger(**best_logger_kwargs)
        
        config = locals()
        config["env"] = str(config["env"])
        config["self"] = "SAC"
        self.logger.save_config(config)
        self.best_logger.save_config(config)
        

        buff_device = torch.device("cpu")
        comp_device = torch.device("cpu")

        if torch.cuda.is_available():
            if use_gpu_buffer:
                buff_device = torch.device("cuda")
                self.logger.log("\nUsing GPU replay buffer\n")

            if use_gpu_computation:
                comp_device = torch.device("cuda")
                self.logger.log("\nUsing GPU computaion\n")
        else:
            self.logger.log("\nGPU unavailable\n")

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.env = env
        self.test_env = env
        self.gamma = gamma
        self.polyak = polyak
        self.num_test_episodes = num_test_episodes
        self.max_ep_len = max_ep_len
        self.start_steps = start_steps
        self.update_after = update_after
        self.update_every = update_every
        self.num_updates = num_updates
        self.save_freq = save_freq
        self.batch_size = batch_size
        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.ent_coef_optimizer = None

        act_dim = env.action_space.shape[0]

        # Action limit for clamping: critically, assumes all dimensions share the same bound!
        self.act_limit = env.action_space.high[0]

        # Create actor-critic module and target networks
        self.ac = actor_critic(env.observation_space,
                               env.action_space,
                               device=comp_device,
                               **ac_kwargs)
        self.ac_targ = deepcopy(self.ac)

        # Freeze target networks with respect to optimizers (only update via polyak averaging)
        for p in self.ac_targ.parameters():
            p.requires_grad = False

        # List of parameters for both Q-networks (save this for convenience)
        self.q_params = itertools.chain(self.ac.q1.parameters(),
                                        self.ac.q2.parameters())

        # Experience buffer
        self.rb = replay_buffer(obs_space=env.observation_space,
                                act_dim=act_dim,
                                env=env,
                                device=buff_device,
                                **rb_kwargs)

        # Count variables (protip: try to get a feel for how different size networks behave!)
        var_counts = tuple(
            count_vars(module)
            for module in [self.ac.pi, self.ac.q1, self.ac.q2])
        self.logger.log(
            "\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n" %
            var_counts)

        # Set up optimizers for policy and q-function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=lr)
        self.q_optimizer = Adam(self.q_params, lr=lr)

        # Set up model saving
        self.logger.setup_pytorch_saver(self.ac)
        self.best_logger.setup_pytorch_saver(self.ac)

        # Target entropy is used when learning the entropy coefficient
        if self.target_entropy == "auto":
            # automatically set target entropy if needed
            self.target_entropy = -np.prod(self.env.action_space.shape).astype(np.float32)
        else:
            # Force conversion
            # this will also throw an error for unexpected string
            self.target_entropy = float(self.target_entropy)

        # The entropy coefficient or entropy can be learned automatically
        # see Automating Entropy Adjustment for Maximum Entropy RL section
        # of https://arxiv.org/abs/1812.05905
        if isinstance(self.ent_coef, str) and self.ent_coef.startswith("auto"):
            # Default initial value of ent_coef when learned
            init_value = 1.0
            if "_" in self.ent_coef:
                init_value = float(self.ent_coef.split("_")[1])
                assert init_value > 0.0, "The initial value of ent_coef must be greater than 0"

            # Note: we optimize the log of the entropy coeff which is slightly different from the paper
            # as discussed in https://github.com/rail-berkeley/softlearning/issues/37
            self.log_ent_coef = torch.log(torch.ones(1, device=comp_device) * init_value).requires_grad_(True)
            self.ent_coef_optimizer = torch.optim.Adam([self.log_ent_coef], lr=lr)
            self.ent_coef = torch.tensor(float(init_value)).to(comp_device)
        else:
            # Force conversion to float
            # this will throw an error if a malformed string (different from 'auto')
            # is passed
            self.ent_coef = torch.tensor(float(self.ent_coef)).to(comp_device)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(self, data):
        o, a, r, o2, d = data["obs"], data["act"], data["rew"], data[
            "obs2"], data["done"]

        q1 = self.ac.q1(o, a)
        q2 = self.ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = self.ac.pi(o2)

            # Target Q-values
            q1_pi_targ = self.ac_targ.q1(o2, a2)
            q2_pi_targ = self.ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + self.gamma * (1 - d) * (q_pi_targ -
                                                 self.ent_coef * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = F.mse_loss(q1, backup)
        loss_q2 = F.mse_loss(q2, backup)
        loss_q = 0.5  * (loss_q1 + loss_q2)

        # Useful info for logging
        q_info = dict(q1_vals=q1.detach().cpu().numpy(),
                      q2_vals=q2.detach().cpu().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(self, data):
        o = data["obs"]
        pi, logp_pi = self.ac.pi(o)
        q1_pi = self.ac.q1(o, pi)
        q2_pi = self.ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (self.ent_coef * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(log_pi=logp_pi.detach().cpu().numpy())

        return loss_pi, logp_pi, pi_info

    def compute_loss_ent_coef(self, logp_pi):
        # Important: detach the variable from the graph
        # so we don't change it with other losses
        # see https://github.com/rail-berkeley/softlearning/issues/60
        ent_coef = torch.exp(self.log_ent_coef.detach())
        ent_coef_loss = -(self.log_ent_coef * (logp_pi + self.target_entropy).detach()).mean()
        
        return ent_coef_loss, ent_coef

    def update(self, data):
        # First run one gradient descent step for Q1 and Q2
        self.q_optimizer.zero_grad()
        loss_q, q_info = self.compute_loss_q(data)
        loss_q.backward()
        self.q_optimizer.step()

        # Record things
        self.logger.store(loss_q=loss_q.item(), **q_info)

        # Freeze Q-networks so you don"t waste computational effort
        # computing gradients for them during the policy learning step.
        for p in self.q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        self.pi_optimizer.zero_grad()
        loss_pi, logp_pi, pi_info = self.compute_loss_pi(data)
        loss_pi.backward()
        self.pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in self.q_params:
            p.requires_grad = True

        # Record things
        self.logger.store(loss_pi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(self.ac.parameters(),
                                 self.ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(self.polyak)
                p_targ.data.add_((1 - self.polyak) * p.data)

        # Optimize entropy coefficient, also called
        # entropy temperature or alpha in the paper
        if self.ent_coef_optimizer is not None:
            ent_coef_loss, ent_coef = self.compute_loss_ent_coef(logp_pi)
            self.ent_coef_optimizer.zero_grad()
            ent_coef_loss.backward()
            self.ent_coef_optimizer.step()
            self.ent_coef = ent_coef
            self.logger.store(ent_coef_loss=ent_coef_loss.item())

    def get_action(self, o, deterministic=False):
        a = self.ac.act(o, deterministic)
        return np.clip(a, -self.act_limit, self.act_limit)

    def test_agent(self):
        for j in range(self.num_test_episodes):
            o, d, ep_ret, ep_len = self.test_env.reset(), False, 0, 0
            while not (d or (ep_len == self.max_ep_len)):
                # Take deterministic actions at test time
                o, r, d, i = self.test_env.step(self.get_action(o, True))
                ep_ret += r
                ep_len += 1
            self.logger.store(test_ep_return=ep_ret, test_ep_length=ep_len)
            if "is_success" in i:
                self.logger.store(test_success_rate=i["is_success"])

    def train(self,
              steps_per_epoch,
              epochs,
              stop_return=None,
              stop_success_rate=None,
              abort_after_epoch=None,
              abort_return_threshold=0.1):
        # Prepare for interaction with environment
        total_steps = steps_per_epoch * epochs
        start_time = time.time()
        o, ep_ret, ep_len = self.env.reset(), 0, 0
        self.rb.start_episode()
        test_ep_return = None
        best_test_ep_return = float("-inf")

        # Main loop: collect experience in env and update/log each epoch
        for t in range(total_steps):
            it_start_time = time.time()
            # Until start_steps have elapsed, randomly sample actions
            # from a uniform distribution for better exploration. Afterwards,
            # use the learned policy.
            if t > self.start_steps:
                a = self.get_action(o)
            else:
                a = self.env.action_space.sample()

            # Step the env
            o2, r, d, i = self.env.step(a)
            ep_ret += r
            ep_len += 1

            # Ignore the "done" signal if it comes from hitting the time
            # horizon (that is, when it"s an artificial terminal signal
            # that isn"t based on the agent"s state)
            d = False if ep_len == self.max_ep_len else d

            # Store experience to replay buffer
            self.rb.store(o, a, r, o2, d, i)

            # Super critical, easy to overlook step: make sure to update
            # most recent observation!
            o = o2

            # End of trajectory handling
            if d or (ep_len == self.max_ep_len):
                self.logger.store(ep_return=ep_ret, ep_length=ep_len)
                self.rb.end_episode()
                o, ep_ret, ep_len = self.env.reset(), 0, 0
                self.rb.start_episode()

            # Update handling
            if t >= self.update_after and t % self.update_every == 0:
                for j in range(self.num_updates or self.update_every):
                    batch = self.rb.sample_batch(self.batch_size)
                    self.update(data=batch)

            self.logger.store(iteration_time=time.time() - it_start_time)

            # End of epoch handling
            if (t + 1) % steps_per_epoch == 0:
                epoch = (t + 1) // steps_per_epoch

                # Save model
                if (epoch % self.save_freq == 0) or (epoch == epochs):
                    self.logger.save_state({"env": self.env}, None)

                # Test the performance of the deterministic version of the agent.
                self.test_agent()

                test_ep_return = self.logger.get_stats("test_ep_return")[0]
                # test_success_rate = self.logger.get_stats(
                #     "test_success_rate")[0]

                # Log info about epoch
                self.logger.log_tabular("epoch", epoch)
                if "test_success_rate" in self.logger.epoch_dict:
                    self.logger.log_tabular("test_success_rate",
                                            average_only=True)
                self.logger.log_tabular("ep_return", average_only=True)
                self.logger.log_tabular("test_ep_return", average_only=True)
                self.logger.log_tabular("ep_length", average_only=True)
                self.logger.log_tabular("test_ep_length", average_only=True)
                self.logger.log_tabular("total_timesteps", t)
                self.logger.log_tabular("loss_pi", average_only=True)
                self.logger.log_tabular("loss_q", average_only=True)
                if self.ent_coef_optimizer is not None:
                    self.logger.log_tabular("ent_coef_loss", average_only=True)
                self.logger.log_tabular("ent_coef", self.ent_coef.item())
                self.logger.log_tabular("time_elapsed",
                                        time.time() - start_time)
                self.logger.log_tabular("iteration_time", average_only=True)
                self.logger.dump_tabular()

                if stop_return is not None and test_ep_return >= stop_return:
                    self.logger.log("\nStopping early\n")
                    break

                # if stop_success_rate is not None and test_success_rate >= stop_success_rate:
                #     self.logger.log("\nStopping early\n")
                #     break

                if abort_after_epoch is not None and epoch >= abort_after_epoch and test_ep_return < abort_return_threshold:
                    self.logger.log("\nAborting ineffectivse training\n")
                    break

                if test_ep_return >= best_test_ep_return:
                    best_test_ep_return = test_ep_return
                    self.logger.log("\nSaving best model\n")
                    self.best_logger.save_state({"env": self.env}, None)

        return test_ep_return