import json
import numpy
import numpy.random
import next.utils as utils
from next.apps.AppDashboard import AppDashboard
import matplotlib.pyplot as plt
import mpld3


class MyAppDashboard(AppDashboard):
    def n_positive_history(self, _, butler):
        exp_args = butler.experiment.get(key='args')
        alg_list = exp_args['alg_list']
        alg_labels = [alg['alg_label'] for alg in alg_list]
        history = dict()
        for alg_label in alg_labels:
            history[alg_label] = butler.algorithms.get(alg_label, key='history')

        plt.figure()

        for alg_label, alg_history in history.items():
            n_queries = []
            n_positive = []
            for h in alg_history:
                n_queries.append(h['n_queries'])
                n_positive.append(h['n_positive'])
            plt.plot(n_queries, n_positive, label=alg_label)

        plt.legend(loc='upper left')
        plt.xlabel('# of queries')
        plt.ylabel('# of positives')

        fig = plt.gcf()
        plot = mpld3.fig_to_dict(fig)
        plt.close()
        return plot

    def test_error_multiline_plot(self, app, butler):
        """
        Description: Returns multiline plot where there is a one-to-one mapping lines to
        algorithms and each line indicates the error on the validation set with respect to number of reported answers

        Expected input:
          None

        Expected output (in dict):
          (dict) MPLD3 plot dictionary
        """
        # get list of algorithms associated with project
        args = butler.experiment.get(key='args')
        test_alg_label = args['alg_list'][0]['test_alg_label']

        test_S = butler.queries.get(pattern={'exp_uid': app.exp_uid, 'alg_label': test_alg_label})
        x_min = numpy.float('inf')
        x_max = -numpy.float('inf')
        y_min = numpy.float('inf')
        y_max = -numpy.float('inf')
        list_of_alg_dicts = []

        for algorithm in args['alg_list']:
            alg_label = algorithm['alg_label']
            list_of_log_dict = butler.ell.get_logs_with_filter(app.app_id + ':ALG-EVALUATION', {'exp_uid': app.exp_uid, 'alg_label': alg_label})
            list_of_log_dict = sorted(list_of_log_dict, key=lambda item: utils.str2datetime(item['timestamp']))
            x = []
            y = []
            for item in list_of_log_dict:
                num_reported_answers = item['num_reported_answers']
                Xd = item['X']
                err = 0.5
                if len(test_S) > 0:
                    # compute error rate
                    number_correct = 0.
                    for query in test_S:
                        if 'q' in query:
                            i, j, k = query['q']
                            score = numpy.dot(Xd[j], Xd[j]) - 2 * numpy.dot(Xd[j], Xd[k]) + 2 * numpy.dot(Xd[i], Xd[k]) - numpy.dot(Xd[i], Xd[i])
                            if score > 0:
                                number_correct += 1.0

                    accuracy = number_correct / len(test_S)
                    err = 1.0 - accuracy
                x.append(num_reported_answers)
                y.append(err)
            alg_dict = {'legend_label': alg_label, 'x': x, 'y': y}
            try:
                x_min = min(x_min, min(x))
                x_max = max(x_max, max(x))
                y_min = min(y_min, min(y))
                y_max = max(y_max, max(y))
            except:
                pass
            list_of_alg_dicts.append(alg_dict)

        import matplotlib.pyplot as plt
        import mpld3
        fig, ax = plt.subplots(subplot_kw=dict(axisbg='#EEEEEE'))
        for alg_dict in list_of_alg_dicts:
            ax.plot(alg_dict['x'], alg_dict['y'], label=alg_dict['legend_label'])
        ax.set_xlabel('Number of answered triplets')
        ax.set_ylabel('Error on hold-out set')
        ax.set_xlim([x_min, x_max])
        ax.set_ylim([y_min, y_max])
        ax.grid(color='white', linestyle='solid')
        ax.set_title('Triplet Test Error', size=14)
        legend = ax.legend(loc=2, ncol=3, mode="expand")
        for label in legend.get_texts():
            label.set_fontsize('small')
        plot_dict = mpld3.fig_to_dict(fig)
        plt.close()
        return plot_dict
