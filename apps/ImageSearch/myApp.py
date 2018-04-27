import json
from next.apps.SimpleTargetManager import SimpleTargetManager
from next.apps.App import App, Butler
from next.utils import debug_print, get_app_alg
import random
import numpy as np
import os


ROOF_LABEL_FILE = ''
LOAD_ROOF = False
SAVE_LABELS = False


class MyApp:
    def __init__(self, db):
        self.app_id = 'ImageSearch'
        self.TargetManager = SimpleTargetManager(db)

    def initExp(self, butler, init_algs, args):

        targets = args['targets']['targetset']

        if isinstance(targets, int):
            targets = [{'image': '{}.jpg'.format(i)} for i in xrange(targets)]

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
            seed_target = '{}.jpg'.format(seed_target)

        try:
            seed_i = targets.index({'image': seed_target})
        except ValueError as e:
            debug_print('seed {} not found in targets'.format(seed_target))
            raise e
        args['seed_i'] = seed_i

        self.TargetManager.set_targetset(butler.exp_uid, targets)
        del args['targets']

        alg_args = {'seed_i': seed_i, 'n': n, 'alg_args': args['alg_args']}
        init_algs(alg_args)

        if LOAD_ROOF and os.path.exists(ROOF_LABEL_FILE):
            with open(ROOF_LABEL_FILE) as f:
                roof_labels = json.load(f)
            butler.experiment.set(key='labels', value=roof_labels)
        else:
            butler.experiment.set(key='labels', value=[(seed_i, 1)])
        butler.experiment.set(key='duplicates', value=0)

        return args

    def getQuery(self, butler, alg, args):
        labels = dict(butler.experiment.get(key='labels'))
        alg_id, alg_label, index = alg()
        while index in labels:
            debug_print('duplicate query, calling getQuery again...', color='red')
            duplicates = butler.experiment.increment(key='duplicates')
            debug_print('{} found so far'.format(duplicates), color='red')
            label = labels[index]
            app = App(butler.app_id, butler.exp_uid, butler.db, butler.ell)
            _alg = get_app_alg(butler.app_id, alg_id)
            _butler = Butler(butler.app_id, butler.exp_uid, self.TargetManager, butler.db, butler.ell, alg_label, alg_id)
            app.run_alg(_butler, alg_label, _alg, 'processAnswer', {'index': index,
                                                                    'label': label})
            _, _, index = alg()
            labels = dict(butler.experiment.get(key='labels'))

        target = butler.targets.get_target_item(butler.exp_uid, index)
        return {'target': target, 'index': index}

    def processAnswer(self, butler, alg, args):
        index = args['index']
        label = args['label']
        butler.experiment.append(key='labels', value=(index, label))
        alg_args = {'index': index, 'label': label}
        alg(alg_args)
        if SAVE_LABELS:
            with butler.experiment.memory.lock('save_labels'):
                with open('labels.{}.json'.format(butler.exp_uid), 'w') as f:
                    json.dump(butler.experiment.get(key='labels'), f)
        return args

    def getModel(self, butler, alg, args):
        return alg()

    def chooseAlg(self, butler, alg_list, _):
        return min(alg_list, key=lambda alg: butler.algorithms.get(uid=alg['alg_label'], key='n_responses'))
