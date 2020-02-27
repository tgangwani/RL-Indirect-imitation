import torch
import torch.nn as nn
from i2l.misc.utils import obs_batch_normalize, log_sum_exp, RunningMeanStd

class Discriminator(nn.Module):
    def __init__(self, ob_dim, ac_dim, hidden_dim, device):
        super(Discriminator, self).__init__()

        self.ob_dim = ob_dim
        self.ac_dim = ac_dim
        input_dim = ob_dim + ac_dim

        actv = nn.Tanh
        self.tower = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, hidden_dim), actv(),
            nn.Linear(hidden_dim, 1, bias=False))

        # log(normalization-constant)
        self.logZ = nn.Parameter(torch.ones(1))

        self.input_rms = RunningMeanStd(shape=input_dim)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=3e-4)
        self.device = device
        self.to(device)
        self.train()

    def forward(self):
        raise NotImplementedError()

    def update(self, ac_eval_fn, pq_buffer, rollouts, num_grad_steps):
        self.train()
        pqb_gen = pq_buffer.data_gen_finite(len(pq_buffer)//num_grad_steps)
        policy_data_gen = rollouts.feed_forward_generator(fetch_normalized=False, advantages=None, mini_batch_size=rollouts.num_steps//num_grad_steps)

        loss_val = 0
        n = 0
        for _ in range(num_grad_steps):

            policy_batch = next(policy_data_gen)
            pqb_batch = next(pqb_gen)

            pqb_state, pqb_action, pqb_pretanh_action = pqb_batch
            policy_state, policy_action, policy_log_probs = \
                    policy_batch[0], policy_batch[2], policy_batch[7]

            # obtain log-prob of the expert actions under the current policy
            pqb_state_normalized = obs_batch_normalize(pqb_state, update_rms=False, rms_obj=rollouts.ob_rms)
            with torch.no_grad():
                pqb_log_probs = ac_eval_fn(pqb_state_normalized, rnn_hxs=None, masks=None, action=pqb_action, pretanh_action=pqb_pretanh_action)[1]

            policy_log_probs = policy_log_probs.detach() / self.ac_dim
            pqb_log_probs = pqb_log_probs.detach() / self.ac_dim

            policy_sa = torch.cat([policy_state, policy_action], dim=1)
            pqb_sa = torch.cat([pqb_state, pqb_action], dim=1)

            # normalize inputs
            normalized_sa = obs_batch_normalize(torch.cat([policy_sa, pqb_sa], dim=0), update_rms=True, rms_obj=self.input_rms)
            policy_sa, pqb_sa = torch.split(normalized_sa, [policy_state.size(0), pqb_state.size(0)], dim=0)

            policy_logp = self.tower(policy_sa)
            pqb_logp = self.tower(pqb_sa)

            policy_logq = policy_log_probs + self.logZ.expand_as(policy_log_probs)
            pqb_logq = pqb_log_probs + self.logZ.expand_as(pqb_log_probs)

            policy_log_pq = torch.cat([policy_logp, policy_logq], dim=1)
            policy_log_pq = log_sum_exp(policy_log_pq, dim=1, keepdim=True)

            pqb_log_pq = torch.cat([pqb_logp, pqb_logq], dim=1)
            pqb_log_pq = log_sum_exp(pqb_log_pq, dim=1, keepdim=True)

            policy_loss = -(policy_logq - policy_log_pq).mean(0)
            pqb_loss = -(pqb_logp - pqb_log_pq).mean(0)

            reward_bias = (-torch.cat([policy_logp, pqb_logp], dim=0)).clamp_(min=0).mean(0)
            loss = pqb_loss + policy_loss + 2*reward_bias

            loss_val += loss.item()
            n += 1

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.parameters(), max_norm=10.)
            self.optimizer.step()

        return loss_val / n

    def predict_batch_rewards(self, rollouts):
        obs = rollouts.raw_obs[:-1].view(-1, self.ob_dim)
        acs = rollouts.actions.view(-1, self.ac_dim)
        sa = torch.cat([obs, acs], dim=1)
        sa = obs_batch_normalize(sa, update_rms=False, rms_obj=self.input_rms)

        with torch.no_grad():
            self.eval()
            rewards = self.tower(sa).view(rollouts.num_steps, -1, 1)
            rollouts.rewards.copy_(rewards)
