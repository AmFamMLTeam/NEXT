import numpy as np

from apps.ImageSearch.algs.Linear.Linear import Linear
from apps.ImageSearch.algs.utils import can_fit, get_X, sparse2list
from apps.ImageSearch.algs.models import MarginalRegression
from next.utils import debug_print

import time


class NLogNMarginalLinear(Linear):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        butler.algorithms.set(key='coefs', value=None)
        butler.algorithms.set(key='n_coefs', value=None)
        return super(NLogNMarginalLinear, self).initExp(butler, n, seed_i, alg_args)

    def select_features(self, butler, _):
        X = get_X(butler)
        t0 = time.time()
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        if can_fit(y):
            model = MarginalRegression().fit(X[labeled], y)
            k = int(max(1, len(y) / np.log10(len(y))))
            debug_print('selecting top {} features'.format(k))
            mask = model.topfeatures(k)
            coefs = model.coef_
            coefs[np.logical_not(mask)] = 0
            coefs = sparse2list(coefs)
            butler.algorithms.set(key='coefs', value=coefs)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            butler.algorithms.set(key='select_features_time', value=time.time() - t0)
            return X[:, mask]
        else:
            return X
