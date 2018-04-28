from sklearn.linear_model import LogisticRegressionCV

from apps.ImageSearch.algs.Linear.Linear import Linear

from apps.ImageSearch.algs.utils import can_fit, get_X, sparse2list
from apps.ImageSearch.algs.models import ConstrainedLogisticRegression, constrained_sparsity
import time
import numpy as np
from sklearn.model_selection import GridSearchCV


class LassoLinear(Linear):
    def linear_model(self, cv=3):
        return LogisticRegressionCV(cv=cv, penalty='l1', solver='liblinear')


