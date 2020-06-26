import numpy as np


def generate_uniform_artificial(n_samples, n_actions, n_features):
    contexts = np.random.randint(0, 2, (n_samples, n_features))
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = np.random.randint(0, 10, n_samples)
    return np.asarray(contexts, dtype=float), actions, rewards


def gan_artificial_covertype(wgan, n_samples, n_actions=2, n_features=54):
    contexts = wgan.generate_contexts(n_samples)
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = 0.5 * np.ones(n_samples)
    return np.asarray(contexts, dtype=float), actions, rewards


def gan_artificial_mushroom(wgan, n_samples, n_actions=7, n_features=117):
    contexts = wgan.generate_contexts(n_samples)
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = 0.5 * np.ones(n_samples)
    return np.asarray(contexts, dtype=float), actions, rewards


def gan_artificial_linear(wgan, n_samples, n_actions=8, n_features=10):
    contexts = wgan.generate_contexts(n_samples)
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = 0.5 * np.ones(n_samples)
    return np.asarray(contexts, dtype=float), actions, rewards


def gan_artificial_wheel(wgan, n_samples, n_actions=5, n_features=2):
    contexts = wgan.generate_contexts(n_samples)
    actions = np.random.randint(0, n_actions, n_samples)
    rewards = 0.5 * np.ones(n_samples)
    return np.asarray(contexts, dtype=float), actions, rewards


if __name__ == '__main__':
    print(generate_uniform_artificial(10))
