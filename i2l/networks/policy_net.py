import numpy as np
import torch.nn as nn
from i2l.misc.utils import init
from i2l.misc.distributions import DiagGaussian

class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None, kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if kwargs is None:
            kwargs = {}
        if base is None:
            base = MLPBase

        self.base = base(obs_shape[0], **base_kwargs)
        assert action_space.__class__.__name__ == "Box", "Continous action-space expected."
        self.dist = DiagGaussian(self.base.output_size, action_space.shape[0], **kwargs)

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError()

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            pretanh_action, action = dist.mode()
            # deterministic should only be used for policy "evaluation",
            # at which point we should not be requiring log_prob and entropy
            action_log_probs = dist_entropy = None
        else:
            rsample = dist.rsample()
            if len(rsample) == 2:  # TanhNormal
                pretanh_action, action = rsample
            else:   # Normal
                pretanh_action = action = rsample
            action_log_probs = dist.log_probs(pretanh_action.detach(), action.detach())
            dist_entropy = dist.entropy(pretanh_action, action).mean(0)

        return value, action, pretanh_action, action_log_probs, dist_entropy, rnn_hxs, dist

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, pretanh_action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(pretanh_action, action)
        dist_entropy = dist.entropy(pretanh_action, action).mean(0)

        return value, action_log_probs, dist_entropy, rnn_hxs, dist

class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            raise NotImplementedError()

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        raise NotImplementedError()

class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=64):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, hidden_size)), nn.Tanh(),
            init_(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs
