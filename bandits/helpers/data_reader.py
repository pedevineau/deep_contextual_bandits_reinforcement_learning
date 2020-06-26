import json
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from bandits.helpers.data_reader import DataReader


class DataReader(object):
    """
    Reads pickle created by class Benchmarker so as to process again data (create csv file for export to latex and save figure)
    """

    def __init__(self, path):
        with open(path, 'rb') as handle:
            self.raw_data = pickle.load(handle)
        attributes_to_set = ['test_name', 'num_actions', 'context_dim', 'nb_contexts', 'cum_rew', 'cum_reg',
                             'algo_details']

        for attribute in attributes_to_set:
            setattr(self, attribute, self.raw_data[attribute])
        if "bound method HParams" in self.algo_details:  # for old version of benchmarker
            self.algo_names = [s[s.rfind("'") + 1:] for s in
                               self.algo_details.split("', <bound method HParams.to_json")[:-1]]
        else:
            self.algo_names = [algo_detail[0] for algo_detail in json.loads(self.algo_details)]
            self.hparams = [algo_detail[1] for algo_detail in json.loads(self.algo_details)]

    def display_results(self, save_path=None):
        plt.figure(figsize=(20, 20))
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
            plt.savefig(save_path)
        plt.show()

    def save_final_res_to_tex(self, path):
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
        pd.DataFrame(data=d).to_csv(path, sep='&', line_terminator=' \\\\ \n', header=True, index=False)


if __name__ == '__main__':
    rad = '../../results/'
    name = '_dummy'
    # name = '_NNparams_linear_test2_1_00'
    # name = '_NNparams_linear_test_1_00'
    # name = '_NNparams_linear_test1_0_10'
    obj = DataReader(rad + name + '.pickle')
    # obj.display_table(/)
    obj.save_final_res_to_tex(rad + name + '.csv')
    obj.display_results(rad + name + '.png')
