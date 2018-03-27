import random


class MyAlg:
    def initExp(self, butler, n, seed_i):
        butler.algorithms.set(key='labels', value=[(seed_i, 1)])
        butler.algorithms.set(key='n_positive', value=0)
        butler.algorithms.set(key='n', value=n)
        return True

    def getQuery(self, butler):
        n = butler.algorithms.get(key='n')
        labels = dict(butler.algorithms.get(key='labels'))
        unlabeled = filter(lambda i: i not in labels, xrange(n))
        return random.choice(unlabeled)

    def processAnswer(self, butler, index, label):
        butler.algorithms.append(key='labels', value=(index, label))
        butler.algorithms.increment(key='n_positive', value=int(label == 1))
        return True

    def getModel(self, _):
        return True
