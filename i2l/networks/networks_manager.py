import os
from i2l.misc.utils import expert_gen_inf, zipsame
from i2l.networks.discriminator_model import Discriminator
from i2l.networks.wcritic_model import Wcritic
from i2l.buffers.super_pq import SuperPQ

class NetworksManager:
    """
    Class that handles the airl-discriminator, the wasserstein critic and the priority buffers
    """
    def __init__(self, args, rl_agent):
        obs_dim = rl_agent.obs_dim
        acs_dim = rl_agent.acs_dim
        self.rl_agent = rl_agent

        self.discriminator = Discriminator(obs_dim, acs_dim, hidden_dim=64, device=args.device)
        # Wass-critic to discriminate b/w "state-only" data from the pq-buffer and the expert data
        self.wcritic = Wcritic(obs_dim, hidden_dim=64, device=args.device)

        # high level wrapper around a class that can manage multiple priority queues (if needed)
        self.super_pq = SuperPQ(count=args.num_pqs, capacity=args.pq_capacity, wcritic=self.wcritic, refresh_rate=args.pq_refresh_rate)

        # (infinite) generator to loop over expert states
        src_expert_data_filename = os.path.join(args.experts_dir, "trajs_{}.pickle".format(args.env_name.split('-')[0].lower()))
        self.expert_gen = expert_gen_inf(src_expert_data_filename, args.num_expert_trajs, args.expert_batch_size, subsample_frequency=1, drop_last=True)

        self.optim_params = {k:args.__dict__[k] for k in ['expert_batch_size', 'airl_grad_steps', 'wcritic_grad_steps']}

    def update(self, iter_num):
        if not self.super_pq.is_empty:
            # perform multiple updates of the wcritic classifier using pq-buffer and expert data
            wcritic_loss = self.wcritic.update(iter_num, self.expert_gen, self.super_pq.random_select(),
                    batch_size=self.optim_params['expert_batch_size'], num_grad_steps=self.optim_params['wcritic_grad_steps'])
        else: wcritic_loss = 0.

        completed_trajs = list(self.rl_agent.latest_trajs.values())[:-1]
        assert len(completed_trajs) > 0, "No completed trajectory. Consider increasing args.num_steps"
        completed_trajs_scores = self.wcritic.assign_score(completed_trajs)

        # randomly select one of the pq-buffers, and add completed trajectories to it
        pqb = self.super_pq.random_select(ignore_empty=True)
        for traj, score in zipsame(completed_trajs, completed_trajs_scores):
            pqb.add_traj({**traj, 'score':score})

        # pqb.add_path() does a deepcopy, hence we can free some memory
        for i in list(self.rl_agent.latest_trajs.keys())[:-1]:
            del self.rl_agent.latest_trajs[i]

        # Update the entries in the pq-buffers using the latest critic
        if iter_num and iter_num % self.super_pq.refresh_rate == 0:
            self.super_pq.update()

        # Update rewards with values from the discriminator
        self.discriminator.predict_batch_rewards(self.rl_agent.rollouts)

        # Perform multiple updates of the discriminator classifier using rollouts and pq-buffer
        discriminator_loss = self.discriminator.update(self.rl_agent.action_eval_fn, self.super_pq.random_select(),
                self.rl_agent.rollouts, num_grad_steps=self.optim_params['airl_grad_steps'])

        return [wcritic_loss, discriminator_loss]
