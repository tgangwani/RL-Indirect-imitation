import torch
import torch.nn as nn
from i2l.misc.utils import AddBias

LOG_SIG_MIN = -10
LOG_SIG_MAX = 2

"""
Modify standard PyTorch Normal to be compatible with this code.
"""
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, _, actions: log_prob_normal(
    self, actions).sum(
        -1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self, *args: normal_entropy(self).sum(-1)
FixedNormal.mode = lambda self: self.mean

class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs, tanh_squash, logstd_init):
        super(DiagGaussian, self).__init__()

        self.tanh_squash = tanh_squash
        self.fc_mean = nn.Linear(num_inputs, num_outputs)
        self.logstd = AddBias(float(logstd_init)*torch.ones(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        action_logstd = self.logstd(torch.zeros_like(action_mean))
        action_logstd = torch.clamp(action_logstd, LOG_SIG_MIN, LOG_SIG_MAX)

        if not self.tanh_squash:
            return FixedNormal(action_mean, action_logstd.exp())

        return TanhNormal(action_mean, action_logstd.exp())

class TanhNormal:
    """
    Represent distribution of X where
        Z ~ N(mean, std)
        X ~ tanh(Z)
    """
    def __init__(self, normal_mean, normal_std):
        """
        :param normal_mean: Mean of the wrapped-normal distribution
        :param normal_std: Std of the wrapped-normal distribution
        """
        self._wrapped_normal = FixedNormal(normal_mean, normal_std)

    def log_probs(self, pretanh_actions, actions):
        """
        :param pretanh_actions: Inverse hyperbolic tangent of actions
        """
        if pretanh_actions is None:
            assert False, "Numerical calculation of inverse-tanh could be unstable. To bypass this, provide pre-tanh actions to this function."

        # [warning:] this is the "incorrect" log-prob since we don't include the -log(1-actions^2) term which
        # comes from using tanh as the flow-layer atop the gaussian. Note that for PPO, this term cancels
        # out in the importance sampling ratio
        return self._wrapped_normal.log_probs(None, pretanh_actions)

    def entropy(self, pretanh_actions, actions):
        """
        Return the (incorrect) entropy of the normal-gaussian (pre tanh). Since we do not have a closed analytical form
        for TanhNormal, an alternative is to return a Monte-carlo estimate of -E(log\pi)
        """
        return self._wrapped_normal.entropy()

    def mode(self):
        m = self._wrapped_normal.mode()
        return m, torch.tanh(m)

    def rsample(self):
        """
        Sampling w/ reparameterization.
        """
        z = self._wrapped_normal.rsample()
        return z, torch.tanh(z)

    def sample(self):
        """
        Sampling w/o reparameterization.
        """
        z = self._wrapped_normal.sample().detach()
        return z, torch.tanh(z)
