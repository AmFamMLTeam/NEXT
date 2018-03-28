import random
import numpy as np
import os
from next.utils import debug_print


class BaseAlgorithm:
    def initExp(self, butler, n, seed_i):
        butler.algorithms.set(key='labels', value=[(seed_i, 1)])
        butler.algorithms.set(key='n_positive', value=0)
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='history', value=[])
        return True

    def getQuery(self, butler):
        n = butler.algorithms.get(key='n')
        labels = dict(butler.algorithms.get(key='labels'))
        unlabeled = filter(lambda i: i not in labels, xrange(n))
        return random.choice(unlabeled)

    def processAnswer(self, butler, index, label):
        butler.algorithms.append(key='labels', value=(index, label))
        n_positive = butler.algorithms.increment(key='n_positive', value=int(label == 1))
        n_queries = len(butler.algorithms.get(key='labels'))
        butler.algorithms.append(key='history', value={'n_queries': n_queries,
                                                       'n_positive': n_positive})
        debug_print(butler.algorithms.get(key='history'))
        return True

    def getModel(self, _):
        return True

    def select_features(self, butler, _):
        return get_X(butler)


class NearestNeighbor(BaseAlgorithm):
    def getQuery(self, butler):
        X = self.select_features(butler, {})
        labels = dict(butler.algorithms.get(key='labels'))
        unlabeled = []
        positives = []
        n = butler.algorithms.get(key='n')
        for i in xrange(n):
            if i not in labels:
                unlabeled.append(i)
            elif labels[i] == 1:
                positives.append(i)
        target = random.choice(positives)
        x = X[target]
        X = X[unlabeled]
        dists = np.linalg.norm(X - x, axis=1)
        best = np.argmin(dists)
        return unlabeled[best]


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
