from apps.ImageSearch.algs.LassoNN.LassoNN import LassoNN


class NLassoNN(LassoNN):
    def initExp(self, butler, n, seed_i, alg_args):
        N = alg_args[butler.alg_label].get('N', 1)
        butler.algorithms.set(key='N', value=N)
        return super(NLassoNN, self).initExp(butler, n, seed_i, alg_args)

    def constraint(self, butler):
        N = butler.algorithms.get(key='N')
        labels = dict(butler.algorithms.get(key='labels'))
        n_labels = len(labels)
        return lambda n: n < n_labels/N

