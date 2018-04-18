import numpy as np

from apps.ImageSearch.algs.NearestNeighbor.NearestNeighbor import NearestNeighbor
from apps.ImageSearch.algs.utils import can_fit, get_X, sparse2list
from apps.ImageSearch.algs.models import BestMarginalRegression


class BestMarginalNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i, alg_args):
        butler.algorithms.set(key='coefs', value=None)
        butler.algorithms.set(key='n_coefs', value=None)
        return super(BestMarginalNN, self).initExp(butler, n, seed_i, alg_args)

    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        X = get_X(butler)
        if can_fit(y, 3):
            model = BestMarginalRegression().fit(X[labeled], y)
            mask = model.topfeatures('best')
            coefs = model.coef_
            coefs[np.logical_not(mask)] = 0
            coefs = sparse2list(coefs)
            butler.algorithms.set(key='coefs', value=coefs)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            return X[:, mask]
        else:
            return X
