import glob
import os
from collections import defaultdict, OrderedDict
import numpy as np

import torch
import torch.utils.data
import torch.nn as nn

class OrderedDefaultDict(OrderedDict):
    def __missing__(self, key):
        self[key] = defaultdict(list)
        return self[key]

def zipsame(*seqs):
    L = len(seqs[0])
    assert all(len(seq) == L for seq in seqs[1:])
    return zip(*seqs)

def log_sum_exp(value, dim, keepdim):
    """Numerically stable implementation of the operation:
    value.exp().sum(dim, keepdim).log()
    """
    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    if keepdim is False:
        m = m.squeeze(dim)
    return m + torch.log(torch.sum(torch.exp(value0),
                                   dim=dim, keepdim=keepdim))

def obs_batch_normalize(obs_tnsr, update_rms, rms_obj):
    """
    Use this function for a batch of 1-D tensors only
    """
    obs_tnsr_np = obs_tnsr.numpy()
    if update_rms:
        rms_obj.update(obs_tnsr_np)

    obs_normalized_np = np.clip((obs_tnsr_np - rms_obj.mean) / np.sqrt(rms_obj.var + 1e-8), -10., 10.)
    obs_normalized_tnsr = torch.FloatTensor(obs_normalized_np).view(obs_tnsr.size()).to(obs_tnsr.device)
    return obs_normalized_tnsr

class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias

class RunningMeanStd:
    """
    https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = update_mean_var_count_from_moments(
            self.mean, self.var, self.count, batch_mean, batch_var, batch_count)

def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / (count + batch_count)
    new_var = M2 / (count + batch_count)
    new_count = batch_count + count

    return new_mean, new_var, new_count

def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def cleanup_log_dir(log_dir):
    try:
        os.makedirs(log_dir)
    except OSError:
        files = glob.glob(os.path.join(log_dir, '*.monitor.csv'))
        for f in files:
            try:
                os.remove(f)
            except OSError:
                pass

def expert_gen_inf(filename, num_trajs, batch_sz, subsample_frequency, drop_last):
    """
    A generator which (infinitely) loops over the expert data
    """
    dataset = ExpertDataset(filename, num_trajs, subsample_frequency)
    dloader = torch.utils.data.DataLoader(dataset, batch_size=batch_sz, shuffle=True, drop_last=drop_last)
    gen = iter(dloader)

    while True:
        try:
            *data, = next(gen)
        except StopIteration:
            # restart generator
            gen = iter(dloader)
            *data, = next(gen)
        yield data

class ExpertDataset(torch.utils.data.Dataset):
    def __init__(self, filename, num_trajs, subsample_frequency):
        if filename.endswith('.pickle'):
            all_trajs = ExpertDataset.process_pickle(filename)
        else: raise NotImplementedError()

        # select the best expert trajectory for imitation
        es, idx = torch.sort(all_trajs['rewards'].sum(dim=1), descending=True)
        idx = idx.tolist()[:num_trajs]
        print("Loaded {} expert trajs with average returns {:.2f}".format(num_trajs, es[:num_trajs].mean().item()))

        self.trajs = {}
        start_idx = torch.randint(0, subsample_frequency, size=(num_trajs, ), dtype=torch.int32)

        for k, v in all_trajs.items():
            data = v[idx]

            if k != 'lengths':
                samples = []
                for i in range(num_trajs):
                    samples.append(data[i, start_idx[i]::subsample_frequency])
                self.trajs[k] = torch.stack(samples)
            else:
                self.trajs[k] = data // subsample_frequency

    @staticmethod
    def process_pickle(filename):
        import pickle
        with open(filename, 'rb') as fd:
            data = pickle.load(fd)

        trajs = {}
        trajs.update({'states':torch.from_numpy(data['states'].astype(np.float32))})
        trajs.update({'rewards':torch.from_numpy(data['rewards'].astype(np.float32)).squeeze(dim=2)})
        trajs.update({'lengths':torch.LongTensor(data['lengths'])})
        return trajs

    def __len__(self):
        return self.trajs['lengths'].sum()

    def __getitem__(self, i):
        traj_idx = 0

        # find the trajectory (traj_idx) from which to return the data
        while self.trajs['lengths'][traj_idx] <= i:
            i -= self.trajs['lengths'][traj_idx]
            traj_idx += 1

        return [self.trajs['states'][traj_idx][i]]
