import warnings

import numpy as np
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


class ConstrainedLogisticRegression(LogisticRegression):
    """
    A version of Logistic Regression that keeps a constraint function, for using the constrained_sparsity metric
    """
    def __init__(self, constraint, penalty, class_weight, C=.1):
        self.constraint = constraint
        super(ConstrainedLogisticRegression, self).__init__(penalty=penalty, class_weight=class_weight, C=C)


class MarginalRegression:
    def fit(self, X, y):
        y = np.array(y)
        y[y == 0] = -1
        # scale gives zero mean and unit variance
        X /= np.max(X)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', UserWarning)
            X = preprocessing.scale(X)
        self.coef_ = np.dot(X.T, y)
        return self

    def topfeatures(self, k):
        if not hasattr(self, 'coef_'):
            raise Exception('not fit')
        # sort coefficients highest to lowest
        topk_indices = np.argsort(-np.abs(self.coef_))[:k]
        mask = np.zeros(len(self.coef_), dtype=bool)
        mask[topk_indices] = True
        return mask


def sparsity_score(est, _X, _y):
    sparsity = -np.count_nonzero(est.coef_) / est.coef_.shape[1]
    auc = roc_auc_score(_y, est.decision_function(_X))
    return sparsity + auc


def constrained_sparsity(est, _X, _y):
    if est.constraint(np.count_nonzero(est.coef_)):
        return roc_auc_score(_y, est.decision_function(_X))
    else:
        return -2
