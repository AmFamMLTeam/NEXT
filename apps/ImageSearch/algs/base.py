import random

from apps.ImageSearch.algs.utils import is_locked, get_X
import time

QUEUE_SIZE = 10


class BaseAlgorithm(object):
    def initExp(self, butler, n, seed_i, alg_args):
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
        n_coefs = butler.algorithms.get(key='n_coefs')
        C = butler.algorithms.get(key='C')
        butler.algorithms.append(key='history', value={'n_queries': n_queries,
                                                       'n_positive': n_positive,
                                                       'n_coefs': n_coefs,
                                                       'C': C})
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


