import torch
import torch.nn as nn

class Wcritic(nn.Module):
    def __init__(self, ob_dim, hidden_dim, device):
        super(Wcritic, self).__init__()

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(ob_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1))

        self.warmup = 5
        self.clip = 0.05
        self.optimizer = torch.optim.RMSprop(self.parameters(), lr=5e-5)
        self.device = device
        self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, niter, expert_gen, pq_buffer, batch_size, num_grad_steps):
        """
        Perform multiple updates of the wasserstein classifier using pq-buffer and expert data
        """
        self.train()
        pqb_gen = pq_buffer.data_gen_infinite(min(batch_size, len(pq_buffer)))

        if niter <= self.warmup:
            num_grad_steps *= (self.warmup + 1 - niter)

        loss_val = 0
        n = 0
        for _ in range(num_grad_steps):

            expert_batch = next(expert_gen)
            pqb_batch = next(pqb_gen)

            expert_state = expert_batch[0]
            pqb_state = pqb_batch[0]

            pqb_out = self.tower(pqb_state)
            expert_out = self.tower(expert_state)

            reward_bias = - torch.clamp(pqb_out, max=0).mean(0) - torch.clamp(expert_out, max=0).mean(0)
            loss = pqb_out.mean(0) - expert_out.mean(0) + 2*reward_bias

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()

            # weight clamping to enforce the Lipchitz constraint
            for p in self.parameters():
                p.data.clamp_(-self.clip, self.clip)

        return loss_val / n

    def assign_score(self, trajs):
        """
        Assign scores to trajectories
        """
        traj_scores = []
        for traj in trajs:
            obs = torch.stack(traj['states']).squeeze(dim=1) if isinstance(traj['states'], list) else traj['states']
            rewards = self._single_traj_score(obs)
            traj_scores.append(rewards.sum().item())

        return traj_scores

    def _single_traj_score(self, obs):
        with torch.no_grad():
            self.eval()
            return self.tower(obs)
