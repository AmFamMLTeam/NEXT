import json
from next.apps.SimpleTargetManager import SimpleTargetManager
from next.utils import debug_print
import random


class MyApp:
    def __init__(self, db):
        self.app_id = 'ImageSearch'
        self.TargetManager = SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        targets = args['targets']['targetset']
        n = len(targets)
        args['n'] = n

        seed_target = args['seed_target']
        try:
            seed_i = targets.index({'image': seed_target})
        except ValueError as e:
            debug_print('seed {} not found in targets'.format(seed_target))
            raise e
        args['seed_i'] = seed_i

        self.TargetManager.set_targetset(butler.exp_uid, targets)
        del args['targets']

        alg_args = {'seed_i': seed_i, 'n': n}
        init_algs(alg_args)

        return args

    def getQuery(self, butler, alg, args):
        index = alg()
        target = butler.targets.get_target_item(butler.exp_uid, index)
        return {'target': target, 'index': index}

    def processAnswer(self, butler, alg, args):
        index = args['index']
        label = args['label']
        debug_print('got label {} for index {}'.format(butler.alg_id,
                                                       label,
                                                       index))
        alg_args = {'index': index, 'label': label}
        alg(alg_args)
        return args

    def getModel(self, butler, alg, args):
        return alg()

    def chooseAlg(self, butler, alg_list, _):
        return random.choice(alg_list)
