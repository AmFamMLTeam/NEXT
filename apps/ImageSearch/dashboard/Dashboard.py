from next.apps.AppDashboard import AppDashboard
import matplotlib.pyplot as plt
import mpld3


class MyAppDashboard(AppDashboard):
    def plot_parameter(self, butler, param, ylabel):

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
                if h[param] is not None:
                    n_queries.append(h['n_queries'])
                    n_positive.append(h[param])
            if n_queries:
                plt.plot(n_queries, n_positive, label=alg_label)

        plt.legend(loc='upper left')
        plt.xlabel('# of queries')
        plt.ylabel(ylabel)

        fig = plt.gcf()
        plot = mpld3.fig_to_dict(fig)
        plt.close()
        return plot

    def n_positive_history(self, _, butler):
        return self.plot_parameter(butler, 'n_positive', '# of positives')

    def n_coefs_history(self, _, butler):
        return self.plot_parameter(butler, 'n_coefs', '# of coefficients')

