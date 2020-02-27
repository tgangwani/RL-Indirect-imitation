import argparse
import torch

def get_args():
    parser = argparse.ArgumentParser(description='I2L')
    parser.add_argument(
        '--seed', type=int, default=1, help='random seed')
    parser.add_argument(
        '--env-name',
        default='Hopper-v2',
        help='environment to train on')
    parser.add_argument(
        '--experts-dir',
        default='./experts',
        help='directory that contains expert demonstrations')
    parser.add_argument(
        '--cuda-deterministic',
        action='store_true',
        default=False,
        help="sets flags for determinism when using CUDA (potentially slow!)")
    parser.add_argument(
        '--log-dir',
        default='/tmp/gym/',
        help='directory to save agent logs (default: /tmp/gym)')
    parser.add_argument(
        '--config-file',
        default=None,
        help='optional yaml file with configuration parameters')

    args = parser.parse_args()
    if args.config_file:
        import yaml
        data = yaml.load(open(args.config_file))
        delattr(args, 'config_file')
        arg_dict = args.__dict__
        for key, value in data.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    args.cuda = args.cuda and torch.cuda.is_available()
    return args
