import numpy as np
from sklearn.model_selection import GridSearchCV
import time

from apps.ImageSearch.algs.NearestNeighbor.NearestNeighbor import NearestNeighbor
from apps.ImageSearch.algs.utils import can_fit, get_X, sparse2list
from apps.ImageSearch.algs.models import ConstrainedLogisticRegression, sparsity_score, constrained_sparsity

from next.utils import debug_print


class LassoNN(NearestNeighbor):
    def initExp(self, butler, n, seed_i, alg_args):
        butler.algorithms.set(key='C', value=.1)
        butler.algorithms.set(key='coefs', value=None)
        return super(LassoNN, self).initExp(butler, n, seed_i, alg_args)

    @property
    def scoring(self):
        return constrained_sparsity

    def select_features(self, butler, _):
        X = get_X(butler)
        t0 = time.time()
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        y = [labels.get(k) for k in labeled]
        if can_fit(y, 2):
            constraint = self.constraint(butler)
            C = butler.algorithms.get(key='C')
            Cs = [C*2**n for n in xrange(-2, 3)] + [.1*2**n for n in xrange(-2, 3)]
            Cs = list(set(Cs))
            cv = min(sum(y), 3)
            search = GridSearchCV(ConstrainedLogisticRegression(constraint=constraint, penalty="l1", class_weight="balanced"), cv=cv, param_grid={"C": Cs}, scoring=self.scoring, refit=True)
            model = search.fit(X[labeled], y).best_estimator_
            self.coefs = model.coef_
            sparse_coefs = sparse2list(self.coefs)
            butler.algorithms.set(key='coefs', value=sparse_coefs)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(self.coefs))
            self.C = model.get_params()["C"]
            mask = np.ravel(model.coef_.astype(bool))
            butler.algorithms.set(key='select_features_time', value=time.time() - t0)
            return X[:, mask]
        else:
            return X

    def constraint(self, butler):
        """
        no contraint for vanilla Lasso
        """
        debug_print('using no constraint for {}'.format(butler.alg_label))
        return lambda _: True

