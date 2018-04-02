import numpy as np

from apps.ImageSearch.algs.NearestNeighbor.NearestNeighbor import NearestNeighbor
from apps.ImageSearch.algs.utils import can_fit, get_X
from apps.ImageSearch.algs.models import MarginalRegression


class MarginalNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        return super(MarginalNN, self).initExp(butler, n, seed_i, alg_args)

    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        X = get_X(butler)
        if can_fit(y):
            N = butler.algorithms.get(key='N')
            mask = MarginalRegression().fit(X[labeled], y).topfeatures(len(y) // N)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            return X[:, mask]
        else:
            return X
