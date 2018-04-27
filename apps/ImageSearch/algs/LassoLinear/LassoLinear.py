from sklearn.linear_model import LogisticRegressionCV

from apps.ImageSearch.algs.Linear.Linear import Linear


class LassoLinear(Linear):
    def linear_model(self, cv=3):
        return LogisticRegressionCV(cv=cv, penalty='l1')

