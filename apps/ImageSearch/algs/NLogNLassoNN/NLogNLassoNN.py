from apps.ImageSearch.algs.LassoNN.LassoNN import LassoNN
from apps.ImageSearch.algs.models import constrained_sparsity
import numpy as np

from next.utils import debug_print


class NLogNLassoNN(LassoNN):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        return super(NLogNLassoNN, self).initExp(butler, n, seed_i, alg_args)

    @property
    def scoring(self):
        return constrained_sparsity

    def constraint(self, butler):
        labels = dict(butler.algorithms.get(key='labels'))
        n_labels = len(labels)
        debug_print('using n_coefs < {}/log({}) restraint for {}'.format(n_labels, n_labels, butler.alg_label))
        return lambda n: n < n_labels/np.log10(n_labels)

