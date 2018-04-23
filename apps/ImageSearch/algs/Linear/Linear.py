import random
import time

import numpy as np

from apps.ImageSearch.algs.base import BaseAlgorithm, QUEUE_SIZE
from apps.ImageSearch.algs.utils import is_locked, can_fit
from next.utils import debug_print

from sklearn.linear_model import LogisticRegressionCV


class Linear(BaseAlgorithm):
    def linear_model(self, cv=3):
        return LogisticRegressionCV(cv=cv)

    def fill_queue(self, butler, args):
        if is_locked(butler.algorithms.memory.lock('fill_queue')):
            debug_print('fill_queue is running already')
            return
        try:
            queue = butler.algorithms.get(key='queue')
        except AttributeError:
            debug_print('couldn\'t fill queue, experiment doesn\'t exist yet?')
            return
        if len(queue) > len(args['queue']):
            debug_print('fill_queue called already')
            return
        with butler.algorithms.memory.lock('fill_queue'):
            debug_print('filling queue')
            X = self.select_features(butler, {})
            d = X.shape[1]
            labels = dict(butler.algorithms.get(key='labels'))
            n = butler.algorithms.get(key='n')
            y = []
            unlabeled = []
            positives = []
            labeled = []
            for i in xrange(n):
                if i not in labels:
                    unlabeled.append(i)
                else:
                    labeled.append(i)
                    y.append(labels[i])
                    if labels[i] == 1:
                        positives.append(i)
            if can_fit(y, 2):
                cv = min(3, sum(y))
                model = self.linear_model(cv=cv)
                model = model.fit(X[labeled], y)
                dists = np.dot(X[unlabeled], np.ravel(model.coef_))
                dists = np.argsort(-dists)
            else:
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
                dists = np.linalg.norm(X - x, axis=1)
                debug_print('took {}s'.format(time.time() - t0))
                dists = np.argsort(dists)
            queries = butler.algorithms.get(key='queries') - butler.algorithms.get(key='last_filled')
            queue_size = max(QUEUE_SIZE, queries * 2)
            self.set_queue(butler, [unlabeled[i] for i in dists[:queue_size]])
            butler.algorithms.set(key='last_filled', value=butler.algorithms.get(key='queries'))

    def constraint(self, butler):
        """
        This should return a function that is of the form:
            f(n_coefficients: int) -> bool
        That is, takes the number of features used by the model and returns whether that should be included as a possible model.
        """
        raise NotImplementedError
