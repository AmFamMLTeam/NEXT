import numpy as np

from apps.ImageSearch.algs.NearestNeighbor.NearestNeighbor import NearestNeighbor
from apps.ImageSearch.algs.utils import can_fit, get_X, sparse2list
from apps.ImageSearch.algs.models import MarginalRegression


class NLogNMarginalNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        butler.algorithms.set(key='coefs', value=None)
        butler.algorithms.set(key='n_coefs', value=None)
        return super(NLogNMarginalNN, self).initExp(butler, n, seed_i, alg_args)

    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        X = get_X(butler)
        if can_fit(y):
            N = butler.algorithms.get(key='N')
            model = MarginalRegression().fit(X[labeled], y)
            mask = model.topfeatures(len(y) // np.log10(len(y)))
            coefs = model.coef_
            coefs[np.logical_not(mask)] = 0
            coefs = sparse2list(coefs)
            butler.algorithms.set(key='coefs', value=coefs)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            return X[:, mask]
        else:
            return X
