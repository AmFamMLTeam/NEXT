import random

import numpy as np

from apps.ImageSearch.algs.MarginalNN.MarginalNN import MarginalNN
from apps.ImageSearch.algs.utils import can_fit, get_X
from apps.ImageSearch.algs.models import MarginalRegression

import time


class MarginalPlusNN(MarginalNN):
    def select_features(self, butler, _):
        X = get_X(butler)
        t0 = time.time()
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
        if can_fit(y):
            N = butler.algorithms.get(key='N')
            mask = MarginalRegression().fit(X[sample], y).topfeatures(len(y) // N)
            butler.algorithms.set(key='n_coefs', value=np.count_nonzero(mask))
            butler.algorithms.set(key='select_features_time', value=time.time() - t0)
            return X[:, mask]
        else:
            return X
