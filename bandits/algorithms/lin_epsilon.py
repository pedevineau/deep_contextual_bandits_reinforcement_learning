from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from bandits.core.bandit_algorithm import BanditAlgorithm


class LinEpsilon(BanditAlgorithm):

    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        self.n_a = self.hparams.num_actions
        self.n_d = self.hparams.context_dim
        self.lam = self.hparams.lam
        self.eps = self.hparams.eps

        self.a = np.concatenate(tuple([np.eye(self.n_d)[np.newaxis, :, :] for i in range(self.n_a)]), axis=0) * self.lam
        self.inv_a = np.concatenate(tuple([np.eye(self.n_d)[np.newaxis, :, :] for i in range(self.n_a)]),
                                    axis=0) / self.lam

        self.b = np.zeros((self.n_a, self.n_d))

        self.theta = np.zeros((self.n_a, self.n_d))

        # self.hparams.initial_pulls

    def action(self, context):
        """

        Args:
          context: Context for which the action need to be chosen.

        Returns:
          action: Selected action for the context.
        """

        # Round robin until each action has been selected "initial_pulls" times
        # if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
        #   return self.t % self.hparams.num_actions
        if np.random.random() < self.eps:
            return np.random.randint(self.n_a)

        vals = np.array([
            np.dot(self.theta[i], context)
            for i in range(self.n_a)])

        return np.argmax(vals)

    def update(self, context, action, reward):
        """Updates action posterior using the linear Bayesian regression formula.

        Args:
          context: Last observed context.
          action: Last observed action.
          reward: Last observed reward.
        """

        self.a[action] = self.a[action] + np.tensordot(context, context, axes=0)
        self.inv_a[action] = np.linalg.inv(self.a[action])
        self.b[action] = self.b[action] + reward * context
        self.theta[action] = np.dot(self.inv_a[action], self.b[action])
