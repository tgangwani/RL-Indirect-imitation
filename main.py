import sys
import os
import time
import random
from collections import deque

import numpy as np
import torch

from i2l.misc.utils import cleanup_log_dir
from i2l.misc.arguments import get_args
from i2l.networks.networks_manager import NetworksManager
from i2l.rl.rl_agent import RLAgent

def setup(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    cleanup_log_dir(log_dir)
    torch.set_num_threads(1)
    args.device = torch.device("cuda:0" if args.cuda else "cpu")

def main():
    args = get_args()
    args.num_processes = 1  # future work: (to run more than 1 env in parallel, code needs modifications)

    print("== Starting I2L with the following parameters == ")
    print(vars(args))
    setup(args)

    rl_agent = RLAgent(args)
    manager = NetworksManager(args, rl_agent)

    episode_rewards = deque(maxlen=50)
    episode_lengths = deque(maxlen=50)

    start = time.time()
    num_updates = int(args.num_env_steps) // args.num_steps // args.num_processes

    for j in range(num_updates):

        # collect agent-environment interaction data
        rl_agent.collect_rollout_batch(episode_rewards, episode_lengths)

        # update wasserstein critic, discriminator, and priority buffer
        wcritic_loss, discriminator_loss = manager.update(j)

        # update actor-critic parameters with PPO
        value_loss, action_loss, dist_entropy = rl_agent.update()

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()

            print("Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean length {:.1f}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f} \n Wcritic_loss: {:.2f}, Discriminator_loss: {:.2f}, Entropy: {:.2f}, Value_loss: {:.2f}, Action_loss: {:.2f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_lengths), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), wcritic_loss, discriminator_loss, dist_entropy, value_loss, action_loss))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
