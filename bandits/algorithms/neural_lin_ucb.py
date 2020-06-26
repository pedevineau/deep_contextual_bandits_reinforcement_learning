from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from bandits.algorithms.neural_bandit_model import NeuralBanditModel
from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset


class NeuralLinUCB(BanditAlgorithm):

    def __init__(self, name, hparams, optimizer='RMS'):

        self.name = name
        self.hparams = hparams
        self.n_a = self.hparams.num_actions
        self.n_d = self.hparams.layer_sizes[-1]
        self.alpha = self.hparams.alpha
        self.lam = self.hparams.lam

        self.a = np.concatenate(tuple([np.eye(self.n_d)[np.newaxis, :, :] for i in range(self.n_a)]), axis=0) * self.lam
        self.inv_a = np.concatenate(tuple([np.eye(self.n_d)[np.newaxis, :, :] for i in range(self.n_a)]),
                                    axis=0) / self.lam

        self.b = np.zeros((self.n_a, self.n_d))

        self.theta = np.zeros((self.n_a, self.n_d))

        # Params for BNN

        self.update_freq_nn = hparams.training_freq_network

        self.t = 0
        self.optimizer_n = optimizer

        self.num_epochs = hparams.training_epochs
        self.data_h = ContextualDataset(hparams.context_dim,
                                        hparams.num_actions,
                                        bootstrap=getattr(hparams, 'bootstrap', None),
                                        intercept=False)

        self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))

    def action(self, context):
        """

        Args:
          context: Context for which the action need to be chosen.

        Returns:
          action: Selected action for the context.
        """

        with self.bnn.graph.as_default():
            c = context.reshape((1, self.hparams.context_dim))
            z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c}).flatten()

        vals = np.array([
            np.dot(self.theta[i], z_context) + self.alpha * np.sqrt(np.dot(z_context, np.dot(self.inv_a[i], z_context)))
            for i in range(self.n_a)])

        return np.argmax(vals)

    def update(self, context, action, reward):
        """Updates action posterior using the linear Bayesian regression formula.

        Args:
          context: Last observed context.
          action: Last observed action.
          reward: Last observed reward.
        """
        self.t += 1
        self.data_h.add(context, action, reward)

        if self.t % self.update_freq_nn == 0:
            if self.hparams.reset_lr:
                self.bnn.assign_lr()
            self.bnn.train(self.data_h, self.num_epochs)

            new_z = self.bnn.sess.run(self.bnn.nn,
                                      feed_dict={self.bnn.x: self.data_h.contexts})
            contexts = new_z
            actions = np.array(self.data_h.actions)
            rewards = self.data_h.rewards[np.arange(actions.shape[
                                                        0]), actions]  # strange but data_h.rewards is of shape (n_samples, n_actions) so we select actions pulled by model

            self.a = np.dot(contexts.T, contexts) + np.concatenate(
                tuple([np.eye(self.n_d)[np.newaxis, :, :] for i in range(self.n_a)]), axis=0) * self.lam
            self.b = np.concatenate(tuple(
                [np.dot(rewards[actions == action], contexts[actions == action])[np.newaxis, :] for action in
                 range(self.n_a)]), axis=0)
            self.inv_a = np.concatenate(
                tuple([np.linalg.inv(self.a[action])[np.newaxis, :, :] for action in range(self.n_a)]), axis=0)
            self.theta = np.concatenate(
                tuple([np.dot(self.inv_a[action], self.b[action])[np.newaxis, :] for action in range(self.n_a)]),
                axis=0)

        else:
            c = context.reshape((1, self.hparams.context_dim))
            z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c}).flatten()

            self.a[action] = self.a[action] + np.tensordot(z_context, z_context, axes=0)
            self.inv_a[action] = np.linalg.inv(self.a[action])
            self.b[action] = self.b[action] + reward * z_context
            self.theta[action] = np.dot(self.inv_a[action], self.b[action])
