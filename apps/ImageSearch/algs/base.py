import random
import numpy as np
import os
from next.utils import debug_print
import time
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def is_locked(lock):
    """For use on a redis-py lock. Returns True if locked, False if not."""
    if lock.acquire(blocking=False):
        lock.release()
        return False
    else:
        return True


QUEUE_SIZE = 10


class BaseAlgorithm(object):
    def initExp(self, butler, n, seed_i):
        butler.algorithms.set(key='labels', value=[(seed_i, 1)])
        butler.algorithms.set(key='n_positive', value=0)
        butler.algorithms.set(key='n_responses', value=0)
        butler.algorithms.set(key='n', value=n)
        butler.algorithms.set(key='history', value=[])
        butler.algorithms.set(key='queue', value=[])
        butler.algorithms.set(key='queries', value=0)
        butler.algorithms.set(key='last_filled', value=0)
        butler.job('fill_queue', {'queue': []})
        return True

    def getQuery(self, butler):
        butler.algorithms.increment(key='queries')
        query = None
        while query is None:
            try:
                query = butler.algorithms.pop(key='queue')
            except IndexError:
                time.sleep(.05)
        if not is_locked(butler.algorithms.memory.lock('fill_queue')):
            butler.job('fill_queue', {'queue': butler.algorithms.get(key='queue')})
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

    def set_queue(self, butler, queries):
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
            queries = random.sample(unlabeled, QUEUE_SIZE)
            self.set_queue(butler, queries)


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
            debug_print('computing distances')
            t0 = time.time()
            dists = np.linalg.norm(X - x, axis=1)
            debug_print('took {}s'.format(time.time() - t0))
            dists = np.argsort(dists)
            queries = butler.algorithms.get(key='queries') - butler.algorithms.get(key='last_filled')
            queue_size = max(QUEUE_SIZE, queries * 2)
            self.set_queue(butler, dists[:queue_size])
            butler.algorithms.set(key='last_filled', value=butler.algorithms.get(key='queries'))


class NLogisticRegression(LogisticRegression):
    """
    A version of Logistic Regression that keeps track of an extra parameter N (used for maximum_sparsity scoring) and how many examples were used to train it.
    """
    def __init__(self, N, penalty, class_weight, C=.1):
        self.N = N
        super().__init__(penalty=penalty, class_weight=class_weight, C=C)

    def fit(self, _X, _y, sample_weight=None):
        self.n_trained = len(_y)
        super().fit(_X, _y, sample_weight)


def can_fit(_y):
    return _y.count(1) > 1 and _y.count(0) > 1


def sparsity_score(est, _X, _y):
    sparsity = -np.count_nonzero(est.coef_) / est.coef_.shape[1]
    auc = roc_auc_score(_y, est.decision_function(_X))
    return sparsity + auc


def maximum_sparsity(est, _X, _y):
    if np.count_nonzero(est.coef_) > (est.n_trained / est.N):
        return -2
    else:
        return roc_auc_score(_y, est.decision_function(_X))


class LassoNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i):
        butler.algorithms.set(key='C', value=.1)
        return super(LassoNN, self).initExp(butler, n, seed_i)

    @property
    def scoring(self):
        return sparsity_score

    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        X = get_X(butler)
        if can_fit(y):
            C = butler.algorithms.get(key='C')
            Cs = [C*2**n for n in xrange(-2, 3)] + [.1*2**n for n in xrange(-2, 3)]
            Cs = list(set(Cs))
            search = GridSearchCV(LogisticRegression(penalty="l1", class_weight="balanced"), param_grid={"C": Cs}, scoring=self.scoring, refit=True)
            model = search.fit(X[labeled], y).best_estimator_
            self.coefs = model.coef_
            self.C = model.get_params()["C"]
            mask = np.ravel(model.coef_.astype(bool))
            return X[:, mask]
        else:
            return X

class NLassoN(NearestNeighbor):
    def initExp(self, butler, n, seed_i):
        pass

    @property
    def scoring(self):
        return None


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
