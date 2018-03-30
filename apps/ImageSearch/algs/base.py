import random
import numpy as np
import os
from next.utils import debug_print
import time


def is_locked(lock):
    """For use on a redis-py lock. Returns True if locked, False if not."""
    if lock.acquire(blocking=False):
        lock.release()
        return False
    else:
        return True


QUEUE_SIZE = 10
FILL_EVERY = 5


class BaseAlgorithm:
    def initExp(self, butler, n, seed_i):
        butler.algorithms.set(key='labels', value=[(seed_i, 1)])
        butler.algorithms.set(key='n_positive', value=0)
        butler.algorithms.set(key='n_responses', value=0)
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='history', value=[])
        butler.algorithms.set(key='queue', value=[])
        butler.algorithms.set(key='pops', value=0)
        butler.job('fill_queue', {'k': QUEUE_SIZE})
        return True

    def getQuery(self, butler):
        debug_print('queue for {} has len={}'.format(butler.alg_label,
                                                     len(butler.algorithms.get(key='queue'))))
        query = None
        while query is None:
            try:
                query = butler.algorithms.pop(key='queue')
                pops = butler.algorithms.increment(key='pops', value=1)
                if pops % FILL_EVERY == 0:
                    butler.job('fill_queue', {'k': QUEUE_SIZE})
            except IndexError:
                time.sleep(.1)
        return query

    def processAnswer(self, butler, index, label):
        butler.algorithms.increment(key='n_responses', value=1)
        butler.algorithms.append(key='labels', value=(index, label))
        n_positive = butler.algorithms.increment(key='n_positive', value=int(label == 1))
        n_queries = len(butler.algorithms.get(key='labels'))
        butler.algorithms.append(key='history', value={'n_queries': n_queries,
                                                       'n_positive': n_positive})
        return True

    def getModel(self, _):
        return True

    def select_features(self, butler, _):
        return get_X(butler)

    def append_queries(self, butler, queries):
        butler.algorithms.set(key='queue', value=[])
        for q in queries:
            butler.algorithms.append(key='queue', value=q)

    def fill_queue(self, butler, args):
        if is_locked(butler.algorithms.memory.lock('fill_queue')):
            return
        with butler.algorithms.memory.lock('fill_queue'):
            n = butler.algorithms.get(key='n')
            labels = dict(butler.algorithms.get(key='labels'))
            unlabeled = filter(lambda i: i not in labels, xrange(n))
            queries = random.sample(unlabeled, args.get('k', QUEUE_SIZE))
            self.append_queries(butler, queries)


class NearestNeighbor(BaseAlgorithm):
    def fill_queue(self, butler, args):
        if is_locked(butler.algorithms.memory.lock('fill_queue')):
            return
        with butler.algorithms.memory.lock('fill_queue'):
            X = self.select_features(butler, {})
            labels = dict(butler.algorithms.get(key='labels'))
            unlabeled = []
            positives = []
            n = butler.algorithms.get(key='n')
            debug_print('iterating through n')
            t0 = time.time()
            for i in xrange(n):
                if i not in labels:
                    unlabeled.append(i)
                elif labels[i] == 1:
                    positives.append(i)
            debug_print('took {}s'.format(time.time() - t0))
            target = random.choice(positives)
            x = X[target]
            X = X[unlabeled]
            debug_print('computing distances')
            t0 = time.time()
            dists = np.linalg.norm(X - x, axis=1)
            debug_print('took {}s'.format(time.time() - t0))
            dists = np.argsort(dists)
            self.append_queries(butler, dists[:args.get('k', QUEUE_SIZE)])


def get_X(butler):
    if not hasattr(butler.db, 'store'):
        butler.db.store = dict()
    if butler.exp_uid not in butler.db.store:
        debug_print('loading features from disk')
        feature_file = butler.experiment.get(key='args')['feature_file']
        feature_file = os.path.join('/', 'next_backend', 'features', feature_file)
        if not feature_file.endswith('.npz'):
            feature_file = feature_file + '.npz'
        butler.db.store[butler.exp_uid] = np.load(feature_file, allow_pickle=False)
        n1 = butler.db.store[butler.exp_uid].shape[0]
        n2 = butler.algorithms.get(key='n')
        if n1 != n2:
            raise ValueError('feature matrix does not match number of targets! ({}!={})'.format(n1, n2))
    debug_print('features have shape: {}'.format(butler.db.store[butler.exp_uid].shape))
    return butler.db.store[butler.exp_uid]
