import random
import time

import numpy as np

from apps.ImageSearch.algs.base import BaseAlgorithm, QUEUE_SIZE
from apps.ImageSearch.algs.utils import is_locked
from next.utils import debug_print


class NearestNeighbor(BaseAlgorithm):
    def fill_queue(self, butler, args):
        if is_locked(butler.algorithms.memory.lock('fill_queue')):
            debug_print('fill_queue is running already')
            return
        queue = butler.algorithms.get(key='queue')
        if len(queue) > len(args['queue']):
            debug_print('fill_queue called already')
            return
        with butler.algorithms.memory.lock('fill_queue'):
            debug_print('filling queue')
            X = self.select_features(butler, {})
            d = X.shape[1]
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
            a, b = np.polyfit([4096*2, 1], [10000, 424924], 1)
            n_sample = int(a*d + b)
            if len(unlabeled) > n_sample:
                debug_print('sampling {} unlabeled examples'.format(n_sample))
                unlabeled = random.sample(unlabeled, n_sample)
            X = X[unlabeled]
            debug_print('computing distances')
            t0 = time.time()
            X = X - x
            dists = np.linalg.norm(X, axis=1)
            debug_print('took {}s'.format(time.time() - t0))
            dists = np.argsort(dists)
            queries = butler.algorithms.get(key='queries') - butler.algorithms.get(key='last_filled')
            queue_size = max(QUEUE_SIZE, queries * 2)
            self.set_queue(butler, [unlabeled[i] for i in dists[:queue_size]])
            butler.algorithms.set(key='last_filled', value=butler.algorithms.get(key='queries'))
