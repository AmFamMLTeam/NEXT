import json
from next.apps.SimpleTargetManager import SimpleTargetManager
from next.utils import debug_print
import random
import numpy as np
import os


class MyApp:
    def __init__(self, db):
        self.app_id = 'ImageSearch'
        self.TargetManager = SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        targets = args['targets']['targetset']

        if isinstance(targets, int):
            targets = [{'image': i} for i in xrange(targets)]

        n = len(targets)
        args['n'] = n

        if 'seed_target' in args:
            seed_target = args['seed_target']
        else:
            debug_print('seed_target not given, picking random one using label_file')
            if 'label_file' not in args:
                raise ValueError('arg "label_file" is required if seed_target not given')
            label_file = os.path.join('/', 'next_backend', 'features', args['label_file'])
            if not label_file.endswith('.npz'):
                label_file = label_file + '.npz'
            y = np.load(label_file, allow_pickle=False)
            seed_target = int(random.choice(np.where(y == 1)[0]))

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
        alg_args = {'index': index, 'label': label}
        alg(alg_args)
        return args

    def getModel(self, butler, alg, args):
        return alg()

    def chooseAlg(self, butler, alg_list, _):
        return min(alg_list, key=lambda alg: butler.algorithms.get(uid=alg['alg_label'], key='n_responses'))
