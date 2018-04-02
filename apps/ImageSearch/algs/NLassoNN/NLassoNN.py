from apps.ImageSearch.algs.LassoNN.LassoNN import LassoNN
from apps.ImageSearch.algs.models import maximum_sparsity


class NLassoNN(LassoNN):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        return super(NLassoNN, self).initExp(butler, n, seed_i, alg_args)

    @property
    def scoring(self):
        return maximum_sparsity
