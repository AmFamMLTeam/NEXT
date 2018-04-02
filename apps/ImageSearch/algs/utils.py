import os
import time

import numpy as np

from next.utils import debug_print


def is_locked(lock):
    """For use on a redis-py lock. Returns True if locked, False if not."""
    if lock.acquire(blocking=False):
        lock.release()
        return False
    else:
        return True


def can_fit(_y):
    return _y.count(1) > 1 and _y.count(0) > 1


def get_X(butler):
    if not hasattr(butler.db, 'store'):
        butler.db.store = dict()
    if butler.exp_uid not in butler.db.store:
        debug_print('loading features from disk')
        t0 = time.time()
        feature_file = butler.experiment.get(key='args')['feature_file']
        feature_file = os.path.join('/', 'next_backend', 'features', feature_file)
        if not feature_file.endswith('.npz'):
            feature_file = feature_file + '.npz'
        butler.db.store[butler.exp_uid] = np.load(feature_file, allow_pickle=False)
        n1 = butler.db.store[butler.exp_uid].shape[0]
        n2 = butler.algorithms.get(key='n')
        if n1 != n2:
            raise ValueError('feature matrix does not match number of targets! ({}!={})'.format(n1, n2))
        debug_print('took {}s'.format(time.time() - t0))
    return butler.db.store[butler.exp_uid]
