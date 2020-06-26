"""Contextual bandit algorithm that selects an action at random."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from bandits.core.bandit_algorithm import BanditAlgorithm


class FixedPolicySampling(BanditAlgorithm):
    """Defines a baseline; returns an action at random with probs given by p."""

    def __init__(self, name, p, hparams):
        """Creates a FixedPolicySampling object.

        Args:
          name: Name of the algorithm.
          p: Vector of normalized probabilities corresponding to sampling each arm.
          hparams: Hyper-parameters, including the number of arms (num_actions).

        Raises:
          ValueError: when p dimension does not match the number of actions.
        """

        self.name = name
        self.p = p
        self.hparams = hparams

        if len(p) != self.hparams.num_actions:
            raise ValueError('Policy needs k probabilities.')

    def action(self, context):
        """Selects an action at random according to distribution p."""
        return np.random.choice(range(self.hparams.num_actions), p=self.p)
