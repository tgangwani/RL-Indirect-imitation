#!/usr/bin/env python

import heapq
import itertools
from copy import deepcopy
import numpy as np
import torch

class BufferEntry():
    """
    Defines the type for each entty of the priority queue
    """
    def __init__(self, r, l, _id):
        self.r = r
        self.l = l
        self._id = _id

    def __eq__(self, other):
        return False

    def __gt__(self, other):
        assert isinstance(other, type(self))
        if self.r > other.r and self.l > other.l: return True
        if other.r > self.r and other.l > self.l: return False
        if self.r/self.l > other.r/other.l: return True
        if (self.r == other.r) and (self.l == other.l) and (self._id > other._id): return True
        return False

class PriorityBuffer():
    def __init__(self, capacity):
        self.capacity = capacity
        self.traj_info = dict()
        self.heap_list = list()
        self.counter = itertools.count()
        self.pointer = 0

    def sift(self, new_scores):
        """
        Rearrange the priority queue when the scores (priorities) of the existing entries change
        """
        for e, new_score in zip(self.traj_info.values(), new_scores):
            old_score = e[0].r

            e[0].r = new_score
            if new_score < old_score:
                heapq._siftdown(self.heap_list, 0, self.heap_list.index(e))
            elif new_score > old_score:
                heapq._siftup(self.heap_list, self.heap_list.index(e))

    def add_traj(self, traj):
        """
        Add a new trajectory to the queue, after checking the necessary conditions
        """

        uid = next(self.counter)

        # Convert lists to tensors
        traj['states'] = torch.stack(traj['states']).squeeze(dim=1)
        traj['actions'] = torch.stack(traj['actions']).squeeze(dim=1)
        traj['pretanh_actions'] = torch.stack(traj['pretanh_actions']).squeeze(dim=1)

        # Create a priority queue entry
        pqe = BufferEntry(traj['score'], len(traj['states']), uid)

        # if at capacity, check if the reward for this traj is greater than the min in the buffer currently
        if len(self.traj_info) == self.capacity:
            min_pqe, *_ = self.heap_list[0]
            if pqe > min_pqe:
                _, min_uid, *_ = heapq.heappop(self.heap_list)
                #print('Traj w/ rew:{:.2f}, len:{} removed!'.format(min_pqe.r, min_pqe.l))
                del self.traj_info[min_uid]  #  clear memory for the expunged entry
            else:
                return

        # success, add to buffer
        #print('Traj w/ rew:{:.2f}, len:{} added!'.format(pqe.r, pqe.l))
        full_entry = [pqe, uid, deepcopy(traj)]  # deepcopy since we delete the traj outside this fn call
        self.traj_info[uid] = full_entry
        heapq.heappush(self.heap_list, full_entry)
        self._sync()

    def _sync(self):
        self.obs = torch.cat([e[2]['states'] for e in self.traj_info.values()])
        self.acs = torch.cat([e[2]['actions'] for e in self.traj_info.values()])
        self.pretanh_acs = torch.cat([e[2]['pretanh_actions'] for e in self.traj_info.values()])
        self.pointer = 0

    @property
    def is_empty(self):
        return len(self.heap_list) == 0

    def _shuffle(self):
        idx = np.arange(self.obs.size(0))
        np.random.shuffle(idx)
        self.obs = self.obs[idx, :]
        self.acs = self.acs[idx, :]
        self.pretanh_acs = self.pretanh_acs[idx, :]

    def data_gen_infinite(self, batch_sz):
        """
        data generator (infinite). It goes over the data infinite times
        """
        g = self.data_gen_finite(batch_sz)
        while True:
            try:
                *ret, = next(g)
            except StopIteration:
                # restart generator
                g = self.data_gen_finite(batch_sz)
                *ret, = next(g)
            yield ret

    def data_gen_finite(self, batch_sz):
        """
        data generator (finite). It goes over the data only once
        """
        self._shuffle()
        self.pointer = 0
        assert len(self) >= batch_sz, "Not enough entries in priority buffer."

        obs, acs, pretanh_acs, done = self._next_mb(batch_sz)
        yield (obs, acs, pretanh_acs)

        while not done:
            obs, acs, pretanh_acs, done = self._next_mb(batch_sz)
            yield (obs, acs, pretanh_acs)

    def __len__(self):
        return self.obs.size(0)

    def _next_mb(self, mb_size):
        """
        Return a mini-batch
        """
        end = self.pointer + mb_size
        obs = self.obs[self.pointer:end, :]
        acs = self.acs[self.pointer:end, :]
        pretanh_acs = self.pretanh_acs[self.pointer:end, :]
        self.pointer = end

        return obs, acs, pretanh_acs, (self.pointer + mb_size > len(self.obs))

    def all_scores(self):
        return [round(e[0].r, 2) for e in self.traj_info.values()]
