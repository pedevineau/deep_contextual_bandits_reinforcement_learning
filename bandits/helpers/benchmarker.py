import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bandits.core.contextual_bandit import run_contextual_bandit


class Benchmarker(object):
    """
    Takes functions that create algos and dataset so as to rerun experiments several times and plot results.
    """

    def __init__(self, algo_protos, dataset_proto, num_actions, context_dim, nb_contexts, test_name):
        self.algo_protos = algo_protos
        self.dataset_proto = dataset_proto
        self.test_name = test_name
        self.num_actions = num_actions
        self.context_dim = context_dim
        self.nb_contexts = nb_contexts
        algos = [algo_proto() for algo_proto in self.algo_protos]
        self.algo_names = [algo.name for algo in algos]
        # self.hparams = [algo.hparams.to_json() for algo in algos]

    def run_experiments(self, iterations=10):
        cum_rew = np.zeros((self.nb_contexts, len(self.algo_protos), iterations))
        cum_reg = np.zeros(cum_rew.shape)

        for iter in range(iterations):
            print(str(iter + 1), '/', str(iterations))
            t_init = time.time()

            dataset, opt_linear = self.dataset_proto()
            print('dataset created')
            opt_rewards, opt_actions = opt_linear

            algos = [algo_proto() for algo_proto in self.algo_protos]
            print('algo ready')

            outcome = run_contextual_bandit(self.context_dim, self.num_actions, dataset, algos)
            h_actions, h_rewards = outcome

            cum_rew[:, :, iter] = np.cumsum(h_rewards, axis=0)
            cum_reg[:, :, iter] = np.cumsum(opt_rewards)[:, np.newaxis] - cum_rew[:, :, iter]

            # print('Iter {} took {} ms'%(iter, time.time()-t_init))

        # if other_results is not None:
        #     self.results = np.concatenate((other_results, results), axis=2)
        # else:
        #     self.results = results
        self.cum_rew = cum_rew
        self.cum_reg = cum_reg

    def save_results(self, path, prefix=''):
        # algos = [algo_proto() for algo_proto in self.algo_protos]
        dic = {
            'test_name': self.test_name,
            'num_actions': self.num_actions,
            'context_dim': self.context_dim,
            'nb_contexts': self.nb_contexts,
            'cum_rew': self.cum_rew,
            'cum_reg': self.cum_reg
            # 'algo_details': json.dumps([(name, hparams) for name, hparams in zip(self.algo_names, self.hparams)])
        }

        with open(path + prefix + '_' + self.test_name + '.pickle', 'wb') as handle:
            pickle.dump(dic, handle)

    def save_final_res_to_tex(self, save_path):
        res = self.cum_reg
        means, stds = np.mean(res, axis=2), np.std(res, axis=2)
        # print
        m = np.max(means)
        cell_text = ['%.2f +/- %.2f' % (100 * mean / m, 100 * std / m) for mean, std in zip(means[-1, :], stds[-1, :])]
        order = np.argsort(means[-1, :])

        cellText = np.array(cell_text)[order][:, np.newaxis],
        colLabels = ['Final Regret'],
        rowLabels = np.array(self.algo_names)[order],
        m2 = np.max(means[-1, :] - means[-100, :])
        easy_names = [
            name.replace('_bs', ' batchsize=').replace('_ls', ' layers=').replace('_RMS', '  RMSprop').replace('_SGD',
                                                                                                               ' SGD')
            for name in self.algo_names]
        d = {'Algorithm': np.array(easy_names)[order], 'Final regret': np.array(cell_text)[order],
             'Last100regrets': np.round((means[-1, :] - means[-100, :]) * 100 / m2)[order]}
        pd.DataFrame(data=d).to_csv(save_path + self.test_name + '.csv', sep='&', line_terminator=' \\\\ \n',
                                    header=True, index=False)

    def display_results(self, save_path=None):
        plt.figure(figsize=(10, 20))
        t = np.arange(self.cum_reg.shape[0])
        res = self.cum_reg
        means, stds = np.mean(res, axis=2), np.std(res, axis=2)
        for i, algo_name in enumerate(self.algo_names):
            mean, std = means[:, i], stds[:, i]
            plt.plot(t, mean, label=algo_name)
            plt.fill_between(t, mean - std, mean + std, alpha=0.3)
        plt.xlabel('Step')
        plt.ylabel('Cumulative regret')
        plt.legend()
        plt.title(self.test_name + ' nb Runs : %i, n_d=%i, n_a=%i, t=%i' % (
        res.shape[2], self.context_dim, self.num_actions, self.nb_contexts))
        if save_path is not None:
            plt.savefig(save_path + self.test_name + '.png')
        plt.show()
