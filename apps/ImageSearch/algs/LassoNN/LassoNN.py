import numpy as np
from sklearn.model_selection import GridSearchCV

from apps.ImageSearch.algs.NearestNeighbor.NearestNeighbor import NearestNeighbor
from apps.ImageSearch.algs.utils import can_fit, get_X
from apps.ImageSearch.algs.models import NLogisticRegression, sparsity_score


class LassoNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i, alg_args):
        butler.algorithms.set(key='C', value=.1)
        return super(LassoNN, self).initExp(butler, n, seed_i, alg_args)

    @property
    def scoring(self):
        return sparsity_score

    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        X = get_X(butler)
        if can_fit(y):
            N = butler.algorithms.get(key='N')
            C = butler.algorithms.get(key='C')
            Cs = [C*2**n for n in xrange(-2, 3)] + [.1*2**n for n in xrange(-2, 3)]
            Cs = list(set(Cs))
            search = GridSearchCV(NLogisticRegression(N=N, penalty="l1", class_weight="balanced"), param_grid={"C": Cs}, scoring=self.scoring, refit=True)
            model = search.fit(X[labeled], y).best_estimator_
            self.coefs = model.coef_
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(self.coefs))
            self.C = model.get_params()["C"]
            mask = np.ravel(model.coef_.astype(bool))
            return X[:, mask]
        else:
            return X