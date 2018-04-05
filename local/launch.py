import argparse
import sys
import yaml
from next.apps.test_utils import initExp
import os

NEXT = os.path.join(os.path.dirname(__file__), '..')

parser = argparse.ArgumentParser()

parser.add_argument('-t', '--targets', type=str, required=True)
parser.add_argument('-X', '--features', type=str, required=True)
parser.add_argument('-a', '--alg_list', type=str, required=True)
parser.add_argument('-s', '--seed', type=str, required=True)


base_config = {
    'app_id': 'ImageSearch',
    'args': {
        'algorithm_management_settings': {'mode': 'custom'},
        'participant_to_algorithm_management': 'one_to_many',
        'instructions': '',
        'debrief': '',
    }
}


def main():
    args = parser.parse_args(sys.argv[1:])
    config = base_config.copy()

    with open(args.alg_list) as f:
        alg_list = yaml.load(f)

    alg_args = dict()

    for alg in alg_list:
        alg_args[alg['alg_label']] = alg.get('args', {})
        if 'args' in alg:
            del alg['args']

    config['args']['alg_list'] = alg_list
    config['args']['alg_args'] = alg_args
    config['args']['feature_file'] = args.features
    config['args']['seed_target'] = args.seed

    with open(args.targets) as f:
        targets = yaml.load(f)

    config['args']['targets'] = {'targetset': targets}

    _, response = initExp(config)

    print('success!')


if __name__ == '__main__':
    main()

