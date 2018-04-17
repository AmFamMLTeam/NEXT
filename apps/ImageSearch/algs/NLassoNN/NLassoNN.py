from apps.ImageSearch.algs.LassoNN.LassoNN import LassoNN

from next.utils import debug_print


class NLassoNN(LassoNN):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        return super(NLassoNN, self).initExp(butler, n, seed_i, alg_args)

    def constraint(self, butler):
        N = butler.algorithms.get(key='N')
        labels = dict(butler.algorithms.get(key='labels'))
        n_labels = len(labels)
        debug_print('using n_coefs < {}/{} for {}'.format(n_labels, N, butler.alg_label))
        return lambda n: n < n_labels/float(N)

