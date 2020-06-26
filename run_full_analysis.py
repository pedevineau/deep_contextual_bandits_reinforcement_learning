"""Contextual bandits simulation.
Code corresponding to:
Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks
for Thompson Sampling, by Carlos Riquelme, George Tucker, and Jasper Snoek.
https://arxiv.org/abs/1802.09127
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import tensorflow as tf
from absl import flags
from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.algorithms.lin_epsilon import LinEpsilon
from bandits.algorithms.lin_ucb import LinUCB
# from bandits.data.data_sampler import sample_mushroom_data
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_lin_ucb import NeuralLinUCB
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.algorithms.parameter_noise_sampling import ParameterNoiseSampling
from bandits.algorithms.posterior_bnn_sampling import PosteriorBNNSampling
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.data.bootstrap_thompson_sampling import gan_artificial_covertype, \
    gan_artificial_mushroom, gan_artificial_linear, gan_artificial_wheel
from bandits.data.environments import *
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
from bandits.data.wasserstein_gans import WGANCovertype
from bandits.data.wasserstein_gans import WGANMushroom
from bandits.helpers.benchmarker import Benchmarker

base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)

flags.DEFINE_string('logdir', base_route, 'Base directory to save output')
FLAGS(sys.argv)

############# STARTS HERE ##############"""
name = "linear"
test_name = "full_analysis_" + name

if name == "linear":
    num_actions = 8
    context_dim = 10
    num_contexts = 1500
    # noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    noise_stds = [1 for i in range(num_actions)]

    wgan = WGANCovertype(context_dim, file="linear")
    wgan.train(epochs=400, batch_size=32, sample_interval=50)
    artificial_data_generator = lambda: gan_artificial_linear(wgan, n_samples=50, n_actions=num_actions)

elif name == "mushroom":
    num_actions = 2
    context_dim = 117
    num_contexts = 1500

    wgan = WGANMushroom(context_dim)
    wgan.train(epochs=2000, batch_size=32, sample_interval=50)
    artificial_data_generator = lambda: gan_artificial_mushroom(wgan, n_samples=50, n_actions=num_actions)

elif name == "wheel":
    num_actions = 5
    context_dim = 2
    num_contexts = 1500
    delta = 0.95
    mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
    std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
    mu_large = 50
    std_large = 0.01

    wgan = WGANCovertype(context_dim, file="wheel")
    wgan.train(epochs=1000, batch_size=32, sample_interval=50)
    artificial_data_generator = lambda: gan_artificial_wheel(wgan, n_samples=50, n_actions=num_actions)


elif name == "covertype":
    num_actions = 7
    context_dim = 54
    num_contexts = 1500

    wgan = WGANCovertype(context_dim)
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)
    artificial_data_generator = lambda: gan_artificial_covertype(wgan, n_samples=50, n_actions=num_actions)
else:
    raise Exception('name not recognized')


def dataset_proto(name=name):
    if name == "linear":
        dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                    num_actions, sigma=noise_stds)
    elif name == "wheel":
        dataset, opt_linear = sample_wheel_bandit_data(num_contexts, delta,
                                                       mean_v, std_v,
                                                       mu_large, std_large)


    elif name == "mushroom":
        mush = Mushrooms(num_contexts=num_contexts)
        dataset = mush.table
        opt_rewards, opt_actions = mush.opts[:, 0], mush.opts[:, 1]
        opt_linear = (opt_rewards, opt_actions)

    elif name == "covertype":
        cov = Covertype(num_contexts=num_contexts)
        dataset = cov.table
        opt_rewards, opt_actions = cov.opts[:, 0], cov.opts[:, 1]
        opt_linear = (opt_rewards, opt_actions)

    return dataset, opt_linear


print(dataset_proto()[0].shape)

# Params for algo templates
hparams = tf.contrib.training.HParams(num_actions=num_actions)

hparams_linear = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             a0=6,
                                             b0=6,
                                             lambda_prior=0.25,
                                             initial_pulls=2)

hparams_rms = tf.contrib.training.HParams(num_actions=num_actions,
                                          context_dim=context_dim,
                                          init_scale=0.3,
                                          activation=tf.nn.relu,
                                          layer_sizes=[50],
                                          batch_size=512,
                                          activate_decay=True,
                                          initial_lr=0.1,
                                          max_grad_norm=5.0,
                                          show_training=False,
                                          freq_summary=1000,
                                          buffer_s=-1,
                                          initial_pulls=2,
                                          optimizer='RMS',
                                          reset_lr=True,
                                          lr_decay_rate=0.5,
                                          training_freq=50,
                                          training_epochs=50,
                                          bootstrap=None)

hparams_rms_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                       context_dim=context_dim,
                                                       init_scale=0.3,
                                                       activation=tf.nn.relu,
                                                       layer_sizes=[50],
                                                       batch_size=512,
                                                       activate_decay=True,
                                                       initial_lr=0.1,
                                                       max_grad_norm=5.0,
                                                       show_training=False,
                                                       freq_summary=1000,
                                                       buffer_s=-1,
                                                       initial_pulls=2,
                                                       optimizer='RMS',
                                                       reset_lr=True,
                                                       lr_decay_rate=0.5,
                                                       training_freq=50,
                                                       training_epochs=50,
                                                       bootstrap=artificial_data_generator)

hparams_rmsb = tf.contrib.training.HParams(num_actions=num_actions,
                                           context_dim=context_dim,
                                           init_scale=0.3,
                                           activation=tf.nn.relu,
                                           layer_sizes=[50],
                                           batch_size=512,
                                           activate_decay=True,
                                           initial_lr=0.1,
                                           max_grad_norm=5.0,
                                           show_training=False,
                                           freq_summary=1000,
                                           buffer_s=-1,
                                           initial_pulls=2,
                                           optimizer='RMS',
                                           reset_lr=True,
                                           lr_decay_rate=0.5,
                                           training_freq=50,
                                           training_epochs=50,
                                           bootstrap=None,
                                           q=3, p=0.95)

hparams_rmsb_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                        context_dim=context_dim,
                                                        init_scale=0.3,
                                                        activation=tf.nn.relu,
                                                        layer_sizes=[50],
                                                        batch_size=512,
                                                        activate_decay=True,
                                                        initial_lr=0.1,
                                                        max_grad_norm=5.0,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        buffer_s=-1,
                                                        initial_pulls=2,
                                                        optimizer='RMS',
                                                        reset_lr=True,
                                                        lr_decay_rate=0.5,
                                                        training_freq=50,
                                                        training_epochs=50,
                                                        q=3, p=0.95,
                                                        bootstrap=artificial_data_generator)

hparams_dropout = tf.contrib.training.HParams(num_actions=num_actions,
                                              context_dim=context_dim,
                                              init_scale=0.3,
                                              use_dropout=True,
                                              keep_prob=0.95,
                                              activation=tf.nn.relu,
                                              layer_sizes=[50],
                                              batch_size=512,
                                              activate_decay=True,
                                              initial_lr=0.1,
                                              max_grad_norm=5.0,
                                              show_training=False,
                                              freq_summary=1000,
                                              buffer_s=-1,
                                              initial_pulls=2,
                                              optimizer='RMS',
                                              reset_lr=True,
                                              lr_decay_rate=0.5,
                                              training_freq=50,
                                              training_epochs=50,
                                              bootstrap=None)

hparams_dropout_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                           context_dim=context_dim,
                                                           init_scale=0.3,
                                                           use_dropout=True,
                                                           keep_prob=0.95,
                                                           activation=tf.nn.relu,
                                                           layer_sizes=[50],
                                                           batch_size=512,
                                                           activate_decay=True,
                                                           initial_lr=0.1,
                                                           max_grad_norm=5.0,
                                                           show_training=False,
                                                           freq_summary=1000,
                                                           buffer_s=-1,
                                                           initial_pulls=2,
                                                           optimizer='RMS',
                                                           reset_lr=True,
                                                           lr_decay_rate=0.5,
                                                           training_freq=50,
                                                           training_epochs=50,
                                                           bootstrap=artificial_data_generator)

hparams_linucb = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             alpha=1,
                                             lam=0.1)

hparams_neural_linucb = tf.contrib.training.HParams(num_actions=num_actions,
                                                    context_dim=context_dim,
                                                    init_scale=0.3,
                                                    activation=tf.nn.relu,
                                                    layer_sizes=[50],
                                                    batch_size=512,
                                                    activate_decay=True,
                                                    initial_lr=0.1,
                                                    max_grad_norm=5.0,
                                                    show_training=False,
                                                    freq_summary=1000,
                                                    buffer_s=-1,
                                                    initial_pulls=2,
                                                    optimizer='RMS',
                                                    reset_lr=True,
                                                    lr_decay_rate=0.5,
                                                    training_freq=50,
                                                    training_epochs=50,
                                                    training_freq_network=50,
                                                    bootstrap=None,
                                                    alpha=1,
                                                    lam=0.1)
hparams_neural_linucb_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                                 context_dim=context_dim,
                                                                 init_scale=0.3,
                                                                 activation=tf.nn.relu,
                                                                 layer_sizes=[50],
                                                                 batch_size=512,
                                                                 activate_decay=True,
                                                                 initial_lr=0.1,
                                                                 max_grad_norm=5.0,
                                                                 show_training=False,
                                                                 freq_summary=1000,
                                                                 buffer_s=-1,
                                                                 initial_pulls=2,
                                                                 optimizer='RMS',
                                                                 reset_lr=True,
                                                                 lr_decay_rate=0.5,
                                                                 training_freq=50,
                                                                 training_epochs=50,
                                                                 training_freq_network=50,
                                                                 bootstrap=artificial_data_generator,
                                                                 alpha=1,
                                                                 lam=0.1)
hparams_neural_linthomson = tf.contrib.training.HParams(num_actions=num_actions,
                                                        context_dim=context_dim,
                                                        init_scale=0.3,
                                                        activation=tf.nn.relu,
                                                        layer_sizes=[50],
                                                        batch_size=512,
                                                        activate_decay=True,
                                                        initial_lr=0.1,
                                                        max_grad_norm=5.0,
                                                        show_training=False,
                                                        freq_summary=1000,
                                                        buffer_s=-1,
                                                        optimizer='RMS',
                                                        reset_lr=True,
                                                        lr_decay_rate=0.5,
                                                        training_freq=50,
                                                        training_epochs=50,
                                                        training_freq_network=50,
                                                        bootstrap=None,
                                                        a0=6,
                                                        b0=6,
                                                        lambda_prior=0.25,
                                                        initial_pulls=2)

hparams_neural_linthomson_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                                     context_dim=context_dim,
                                                                     init_scale=0.3,
                                                                     activation=tf.nn.relu,
                                                                     layer_sizes=[50],
                                                                     batch_size=512,
                                                                     activate_decay=True,
                                                                     initial_lr=0.1,
                                                                     max_grad_norm=5.0,
                                                                     show_training=False,
                                                                     freq_summary=1000,
                                                                     buffer_s=-1,
                                                                     optimizer='RMS',
                                                                     reset_lr=True,
                                                                     lr_decay_rate=0.5,
                                                                     training_freq=50,
                                                                     training_epochs=50,
                                                                     training_freq_network=50,
                                                                     bootstrap=artificial_data_generator,
                                                                     a0=6,
                                                                     b0=6,
                                                                     lambda_prior=0.25,
                                                                     initial_pulls=2)

hparams_pnoise = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             init_scale=0.3,
                                             activation=tf.nn.relu,
                                             layer_sizes=[50],
                                             batch_size=512,
                                             activate_decay=True,
                                             initial_lr=0.1,
                                             max_grad_norm=5.0,
                                             show_training=False,
                                             freq_summary=1000,
                                             buffer_s=-1,
                                             initial_pulls=2,
                                             optimizer='RMS',
                                             reset_lr=True,
                                             lr_decay_rate=0.5,
                                             training_freq=50,
                                             training_epochs=100,
                                             noise_std=0.05,
                                             eps=0.1,
                                             d_samples=300
                                             )

hparams_pnoise_bootstrapped = tf.contrib.training.HParams(num_actions=num_actions,
                                                          context_dim=context_dim,
                                                          init_scale=0.3,
                                                          activation=tf.nn.relu,
                                                          layer_sizes=[50],
                                                          batch_size=512,
                                                          activate_decay=True,
                                                          initial_lr=0.1,
                                                          max_grad_norm=5.0,
                                                          show_training=False,
                                                          freq_summary=1000,
                                                          buffer_s=-1,
                                                          initial_pulls=2,
                                                          optimizer='RMS',
                                                          reset_lr=True,
                                                          lr_decay_rate=0.5,
                                                          training_freq=50,
                                                          training_epochs=100,
                                                          noise_std=0.05,
                                                          eps=0.1,
                                                          d_samples=300,
                                                          bootstrap=artificial_data_generator
                                                          )

hparams_lineps = tf.contrib.training.HParams(num_actions=num_actions,
                                             context_dim=context_dim,
                                             lam=0.1,
                                             eps=0.05)

random_proto = lambda: UniformSampling('Uniform Sampling', hparams)
neural_greedy_proto = lambda: PosteriorBNNSampling('NeuralGreedy', hparams_rms, 'RMSProp')
neural_greedy_proto_bootstrapped = lambda: PosteriorBNNSampling('NeuralGreedy_artificial_data',
                                                                hparams_rms_bootstrapped, 'RMSProp')

bootstrap_proto = lambda: BootstrappedBNNSampling('BootRMS', hparams_rmsb)
bootstrap_proto_bootstrapped = lambda: BootstrappedBNNSampling('BootRMS_artificial_data', hparams_rmsb_bootstrapped)

noise_proto = lambda: ParameterNoiseSampling('ParamNoise', hparams_pnoise)
noise_proto_bootstrapped = lambda: ParameterNoiseSampling('ParamNoise_artificial_data', hparams_pnoise_bootstrapped)

dropout_proto = lambda: PosteriorBNNSampling('Dropout', hparams_dropout, 'RMSProp')
dropout_proto_bootstrapped = lambda: PosteriorBNNSampling('Dropout_artificial_data', hparams_dropout_bootstrapped,
                                                          'RMSProp')

linThompson_proto = lambda: LinearFullPosteriorSampling('linThompson', hparams_linear)
linUCB_proto = lambda: LinUCB('linUCB', hparams_linucb)
linEps_proto = lambda: LinEpsilon('LinEpsilon', hparams_lineps)

neuralLinUCB_proto = lambda: NeuralLinUCB('NeuralLinUCB', hparams_neural_linucb, 'RMSProp')
neuralLinThomson_proto = lambda: NeuralLinearPosteriorSampling('NeuralLinThomson', hparams_neural_linthomson, 'RMSProp')
neuralLinUCB_proto_bootstrapped = lambda: NeuralLinUCB('NeuralLinUCB_artificial_data',
                                                       hparams_neural_linucb_bootstrapped, 'RMSProp')
neuralLinThomson_proto_bootstrapped = lambda: NeuralLinearPosteriorSampling('NeuralLinThomson_artificial_data',
                                                                            hparams_neural_linthomson_bootstrapped,
                                                                            'RMSProp')

algo_protos = [linUCB_proto,
               neuralLinUCB_proto, neuralLinUCB_proto_bootstrapped,
               dropout_proto, dropout_proto_bootstrapped,
               bootstrap_proto, bootstrap_proto_bootstrapped,
               noise_proto, noise_proto_bootstrapped,
               neuralLinThomson_proto, neuralLinThomson_proto_bootstrapped,
               linEps_proto,
               linThompson_proto,
               neural_greedy_proto, neural_greedy_proto_bootstrapped,
               random_proto]

# Run experiments several times save and plot results
benchmarker = Benchmarker(algo_protos, dataset_proto, num_actions, context_dim, nb_contexts=num_contexts,
                          test_name=test_name)

benchmarker.run_experiments(50)
benchmarker.save_results('./results/')
benchmarker.save_final_res_to_tex('./results/')
benchmarker.display_results(save_path='./results/')
