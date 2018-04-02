import argparse
import sys
import os
import yaml
import numpy as np
import time
from multiprocessing import Pool, Queue, Process

from tqdm import tqdm

NEXT = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(NEXT)
from local.launch import launch
from next.apps import test_utils

parser = argparse.ArgumentParser()

parser.add_argument('-q', '--queries', type=int, default=50, help='number of queries per algorithm')
parser.add_argument('-p', '--participants', type=int, default=2, help='number of participants')
parser.add_argument('-y', '--labels', type=str, default='cifar-cats-10-labels.npz', help='filename for labels file')
parser.add_argument('-X', '--features', type=str, default='cifar-fc-cats-10.npz', help='filename for features file')
parser.add_argument('-c', '--concurrency', type=int, default=2, help='number of simultaneous labelers')
parser.add_argument('-d', '--delay', type=int, default=100, help='simulated response delay in ms')
parser.add_argument('-a', '--alg_list', type=str, default='cifar/alg_list.yaml', help='yaml file of alg list')

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

    alg_args = dict()

    with open(args.alg_list) as f:
        alg_list = yaml.load(f)

    for alg in alg_list:
        alg_args[alg['alg_label']] = alg.get('args', {})
        if 'args' in alg:
            del alg['args']

    config['args']['alg_list'] = alg_list
    config['args']['alg_args'] = alg_args
    config['args']['feature_file'] = args.features
    config['args']['label_file'] = args.labels
    y = np.load(os.path.join(NEXT, 'features', args.labels))
    config['args']['targets'] = {'targetset': len(y)}

    _, response = test_utils.initExp(config)
    exp_uid = response['exp_uid']

    n_queries = args.queries * len(config['args']['alg_list'])

    q_per_p = n_queries // args.participants
    p1_q = q_per_p + (n_queries - q_per_p * args.participants)

    results_queue = Queue()
    procs = []

    pargs = []
    for i in range(args.participants):
        arg = dict()
        arg['q'] = p1_q if i == 0 else q_per_p
        arg['y'] = y
        arg['d'] = args.delay
        arg['i'] = i
        arg['exp_uid'] = exp_uid
        arg['results_queue'] = results_queue
        procs.append(Process(target=one_participant, args=(arg,)))
        pargs.append(arg)

    for p in procs:
        p.start()

    for _ in tqdm(xrange(n_queries)):
        results_queue.get()

    for p in procs:
        p.join()

    # pool = Pool(args.concurrency)
    # pool.map(one_participant, pargs)


def one_participant(parg):
    results_queue = parg['results_queue']
    for _ in range(parg['q']):
        response, _ = test_utils.getQuery({'exp_uid': parg['exp_uid'],
                                           'args': {
                                               'participant_uid': parg['i'],
                                               'widget': False
                                           }})
        t0 = time.time()
        test_utils.response_delay(mean=parg['d'] * .001)
        index = response['index']
        query_uid = response['query_uid']
        test_utils.processAnswer({'exp_uid': parg['exp_uid'],
                                  'args': {
                                      'query_uid': query_uid,
                                      'response_time': time.time() - t0,
                                      'label': parg['y'][index],
                                      'index': index
                                  }})
        results_queue.put(True)


if __name__ == '__main__':
    main()
