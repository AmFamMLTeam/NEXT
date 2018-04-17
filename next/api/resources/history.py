from flask_restful import Resource, reqparse

from next.api.resource_manager import ResourceManager
from next.database_client.DatabaseAPI import DatabaseAPI
from next.logging_client.LoggerAPI import LoggerAPI
from next.apps.App import App
db = DatabaseAPI()
ell = LoggerAPI()

resource_manager = ResourceManager()

# Request parser. Checks that necessary dictionary keys are available in a given resource.
# We rely on learningLib functions to ensure that all necessary arguments are available and parsed.

# Custom errors for GET and POST verbs on experiment resource
meta_error = {
    'ExpDoesNotExistError': {
        'message': "No experiment with the specified experiment ID exists.",
        'code': 400,
        'status':'FAIL'
    },
}

meta_success = {
    'code': 200,
    'status': 'OK'
}


class History(Resource):
    def get(self, exp_uid):
        app_id = resource_manager.get_app_id(exp_uid)
        app = App(app_id, exp_uid, db, ell)
        butler = app.butler
        alg_list = butler.experiment.get(key='args')['alg_list']
        history = dict()
        for alg_dict in alg_list:
            alg_label = alg_dict['alg_label']
            alg_history = butler.algorithms.get(alg_label, key='history')
            history[alg_label] = alg_history
        return history
