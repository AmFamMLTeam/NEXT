from sklearn.linear_model import LogisticRegressionCV

from apps.ImageSearch.algs.Linear.Linear import Linear


class LassoLinear(Linear):
    def linear_model(self):
        return LogisticRegressionCV(cv=3, penalty='l1')

