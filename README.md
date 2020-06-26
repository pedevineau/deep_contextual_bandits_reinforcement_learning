# Deep Contextual Bandits

*Date: January 2019*

In this repository, I benchmark different [Deep Reinforcement Learning (Deep RL) ](https://en.wikipedia.org/wiki/Deep_reinforcement_learning) algorithms for the problem of contextual bandits.

### 1. What is the Deep Contextual Bandits problem?

[Contextual Bandits](https://en.wikipedia.org/wiki/Multi-armed_bandit#Constrained_contextual_bandit) is a RL problem without any state where a given context/features vector is given.
In Deep Contextual Bandits, a neural network estimates the reward of an action, given a context.
At each RL-iteration, the action with the highest reward -as estimated by the neural network- is chosen.
Once the action has been performed, the actual reward is received by he agent.
The deep learning model is subsequently retrained, based on this real (ground truth) reward.

### 2. Getting started

First create a ``results`` folder at the same level of the ``README.md`` to run these scripts:

    mkdir results

Two scripts can now be used to reproduce the benchmark results.

The first one is used to compare a variety of models on linear/wheel/covertype/mushrooms datasets (note: change the name attribute in script to select the one you want):

    python run_full_analysis.py

The second one is used to compare the performance of a neural greedy model with different hyperparameters:

    python run_nn_analysis.py

### 3. Sources

Some of the files in this repo come from a fork of a github implementation of the *[Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)* paper, published in
[ICLR](https://iclr.cc/) 2018.
The forked github is available at https://github.com/pedevineau/models.
The other files are mine.

Additions from the original project are :
- LinUCB, neuralLinUCB and Lin Epsilon algorithms
- CovertypeGAN and MushroomGAN for general context vectors and categorical context vectors.
- Use of an artifical data generator in neural network based algorithms
- Custom mushroom and covertype dataset readers
- Benchmarker class : allows to run exxperiments several times, display results in a png, store results in pickle, store algo performance in csv format (csv for later tex export) - DataReader class is the associated class that allows to read again the pickle file and process data once again.

### 4. References
 
Some references:
  - [Deep Bayesian Bandits Showdown: An Empirical
Comparison of Bayesian Deep Networks for Thompson
Sampling](https://arxiv.org/abs/1802.09127)
  - [Bootstrapped Thompson Sampling and Deep Exploration](https://arxiv.org/pdf/1507.00300.pdf)
