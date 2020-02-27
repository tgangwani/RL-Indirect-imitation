import random
from i2l.buffers.priority_buffer import PriorityBuffer

class SuperPQ():
    """
    Class that manages multiple priority queues (buffers)
    """
    def __init__(self, count, capacity, wcritic, refresh_rate):
        self.pq_buffers = []
        self.wcritic = wcritic
        self.refresh_rate = refresh_rate
        for _ in range(count):
            self.pq_buffers.append(PriorityBuffer(capacity))

    @property
    def is_empty(self):
        """
        check if all queues are empty
        """
        check = []
        for pqb in self.pq_buffers:
            check.append(pqb.is_empty)
        return all(check)

    def random_select(self, ignore_empty=False):
        """
        return one queue at random
        """
        candidates = [pqb for pqb in self.pq_buffers if not pqb.is_empty]
        if len(candidates) == 0:
            assert ignore_empty, "All priority buffer are unexpectedly empty"
            return self.pq_buffers[0]

        return random.choice(candidates)

    def update(self):
        """
        Update the scores of existing PQ entries using the latest wcritic
        """
        for i, pqb in enumerate(self.pq_buffers):
            if not pqb.is_empty:
                #print("[id={}] PQ traj scores (before):".format(i), pqb.all_scores())
                trajs = [x[2] for x in pqb.traj_info.values()]
                updated_scores = self.wcritic.assign_score(trajs)
                pqb.sift(updated_scores)
                #print("[id={}] PQ traj scores (after):".format(i), pqb.all_scores())
