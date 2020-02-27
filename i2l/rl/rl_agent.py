import torch

from i2l.rl.ppo import PPO
from i2l.misc.envs import make_vec_envs
from i2l.networks.policy_net import Policy
from i2l.buffers.storage import RolloutStorage
from i2l.misc.utils import OrderedDefaultDict

class RLAgent():
    def __init__(self, args):

        self.envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                             args.log_dir, args.device, False)
        assert len(self.envs.observation_space.shape) == 1 and \
                len(self.envs.action_space.shape) == 1, \
                "Expected flat observation and action spaces. Consider adding a wrapper."
        self.obs_dim = self.envs.observation_space.shape[0]
        self.acs_dim = self.envs.action_space.shape[0]

        self.actor_critic_ntwk = Policy(
            self.envs.observation_space.shape,
            self.envs.action_space,
            kwargs={'tanh_squash': args.tanh_squash, 'logstd_init': args.policy_logstd_init})
        self.actor_critic_ntwk.to(args.device)
        self.action_eval_fn = self.actor_critic_ntwk.evaluate_actions

        self.rl_algo = PPO(
            self.actor_critic_ntwk,
            args.ppo_clip_param,
            args.ppo_epochs,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=args.policy_lr,
            max_grad_norm=args.max_grad_norm)

        self.num_steps = args.num_steps
        self.rollouts = RolloutStorage(self.num_steps, args.num_processes,
                                  self.envs.observation_space.shape, self.envs.action_space,
                                  self.actor_critic_ntwk.recurrent_hidden_state_size,
                                  args.use_gae, args.gamma, args.gae_lambda, args.use_proper_time_limits, args.device)


        self.latest_trajs = OrderedDefaultDict()
        self.num_finished_trajs = 0

        obs = self.envs.reset()
        self.rollouts.raw_obs[0].copy_(obs)
        self.rollouts.normalized_obs[0].copy_(obs)

    def collect_rollout_batch(self, episode_rewards, episode_lengths):
        for step in range(self.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, pretanh_action, action_log_prob, _, recurrent_hidden_states, _ = self.actor_critic_ntwk.act(
                    self.rollouts.normalized_obs[step], self.rollouts.recurrent_hidden_states[step],
                    self.rollouts.masks[step])

            self.latest_trajs[self.num_finished_trajs]['states'].append(self.rollouts.raw_obs[step])
            self.latest_trajs[self.num_finished_trajs]['actions'].append(action)
            self.latest_trajs[self.num_finished_trajs]['pretanh_actions'].append(pretanh_action)

            obs, reward, done, infos = self.envs.step(action)
            reward *= 0.  # enable for imitation learning (do not have access to env. rewards)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])
                    episode_lengths.append(info['episode']['l'])
                    self.num_finished_trajs += 1

            # If done, then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])

            self.rollouts.insert(obs, recurrent_hidden_states, action, pretanh_action,
                    action_log_prob, value, reward, masks, bad_masks)

    def update(self):
        #  Compute GAE and TD-lambda estimates, and then update actor-critic parameters
        with torch.no_grad():
            next_value = self.actor_critic_ntwk.get_value(
                self.rollouts.normalized_obs[-1], self.rollouts.recurrent_hidden_states[-1],
                self.rollouts.masks[-1])
        self.rollouts.compute_adv_tdlam(next_value)

        value_loss, action_loss, dist_entropy = self.rl_algo.update(self.rollouts)
        self.rollouts.after_update()
        return value_loss, action_loss, dist_entropy
