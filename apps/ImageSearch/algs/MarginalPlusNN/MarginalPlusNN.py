import random

import numpy as np

from apps.ImageSearch.algs.MarginalNN.MarginalNN import MarginalNN
from apps.ImageSearch.algs.utils import can_fit, get_X
from apps.ImageSearch.algs.models import MarginalRegression


class MarginalPlusNN(MarginalNN):
    def select_features(self, butler, _):
        labels = dict(butler.algorithms.get(key='labels'))
        labeled = list(labels.keys())
        n_trained = len(labels)
        n_positive = sum(labels.values())
        n_negative = n_trained - n_positive
        n_sample = n_positive - n_negative
        n = butler.algorithms.get(key='n')
        unlabeled = [i for i in xrange(n) if i not in labels]
        unlabeled_sample = []
        if 0 < n_sample <= len(unlabeled):
            unlabeled_sample = random.sample(unlabeled, n_sample)
        sample = labeled + unlabeled_sample
        y = [labels.get(i, 0) for i in sample]
        labeled = list(labels.keys())
        X = get_X(butler)
        if can_fit(y):
            N = butler.algorithms.get(key='N')
            mask = MarginalRegression().fit(X[labeled], y).topfeatures(len(y) // N)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            return X[:, mask]
        else:
            return X
