# Wasserstein GAN. Adapted from base code found at: https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py

from __future__ import print_function, division

import os

import keras.backend as K
import numpy as np
from bandits.data.environments import get_labels_contexts_covertype
from bandits.data.environments import get_labels_contexts_mushroom
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data
# from keras.datasets import mnist
from keras.layers import Input, Dense, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import RMSprop


class WGANCovertype:
    def __init__(self, n_features=54, file="datasets/covtype.data"):
        self.n_features = n_features
        self.n_noise = 100
        self.file = file

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.n_noise,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128, activation="relu", input_dim=self.n_noise))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_features, activation="relu"))

        model.summary()

        noise = Input(shape=(self.n_noise,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Dense(16, input_dim=self.n_features))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        img = Input(shape=(self.n_features,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        # (X_train, _), (_, _) = mnist.load_data()
        self.dirname = os.path.dirname(__file__)
        if self.file == "linear":
            num_actions = 8
            context_dim = 10
            num_contexts = 1500
            noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
            noise_stds = [1 for i in range(num_actions)]
            X_train = sample_linear_data(num_contexts, context_dim,
                                         num_actions, sigma=noise_stds)[0][:, :-8]

        elif self.file == "wheel":
            num_actions = 5
            context_dim = 2
            num_contexts = 1500
            delta = 0.95
            mean_v = [1.0, 1.0, 1.0, 1.0, 1.2]
            std_v = [0.05, 0.05, 0.05, 0.05, 0.05]
            mu_large = 50
            std_large = 0.01
            X_train = sample_wheel_bandit_data(num_contexts, delta,
                                               mean_v, std_v,
                                               mu_large, std_large)[0][:, :-5]

        else:
            _, X_train = get_labels_contexts_covertype(path=os.path.join(self.dirname, self.file))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.n_noise))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                # print("piche", d_loss_real, d_loss_fake)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

        # Plot the progress
        # print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

        # If at save interval => save generated image samples
        # if epoch % sample_interval == 0:
        # 	self.show_predictions(epoch)

    def show_predictions(self, epoch):
        # r, c = 5, 5
        noise = np.random.normal(0, 1, (1, self.n_noise))
        gen = self.generator.predict(noise)

    # print(self.critic.predict(gen))

    def generate_contexts(self, n_samples):
        return np.array([self.generator.predict(np.random.normal(0, 1, (1, self.n_noise)))
                         for k in range(n_samples)])


class WGANMushroom():
    """
    This class is used to take into account the categorical format of mushrooms
    """

    def __init__(self, n_features=117, file="datasets/agaricus-lepiota.data"):
        self.n_features = n_features
        self.n_noise = 100
        self.file = file

        # Following parameter and optimizer set as recommended in paper
        self.n_critic = 5
        self.clip_value = 0.01
        optimizer = RMSprop(lr=0.00005)

        # Build and compile the critic
        self.critic = self.build_critic()
        self.critic.compile(loss=self.wasserstein_loss,
                            optimizer=optimizer,
                            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generated imgs
        z = Input(shape=(self.n_noise,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated images as input and determines validity
        valid = self.critic(img)

        # The combined model  (stacked generator and critic)
        self.combined = Model(z, valid)
        self.combined.compile(loss=self.wasserstein_loss,
                              optimizer=optimizer,
                              metrics=['accuracy'])

    def wasserstein_loss(self, y_true, y_pred):
        return K.mean(y_true * y_pred)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128, activation="relu", input_dim=self.n_noise))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(self.n_features, activation="sigmoid"))

        model.summary()

        noise = Input(shape=(self.n_noise,))
        img = model(noise)

        return Model(noise, img)

    def build_critic(self):

        model = Sequential()

        model.add(Dense(16, input_dim=self.n_features))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(32))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Dense(64, activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(128, activation="relu"))
        model.add(Dropout(0.25))
        # model.add(Flatten())
        model.add(Dense(1, activation="sigmoid"))

        model.summary()

        img = Input(shape=(self.n_features,))
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        self.dirname = os.path.dirname(__file__)
        _, X_train = get_labels_contexts_mushroom(path=os.path.join(self.dirname, self.file))

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            for _ in range(self.n_critic):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Select a random batch of images
                idx = np.random.randint(0, X_train.shape[0], batch_size)
                imgs = X_train[idx]

                # Sample noise as generator input
                noise = np.random.normal(0, 1, (batch_size, self.n_noise))

                # Generate a batch of new images
                gen_imgs = self.generator.predict(noise)

                # Train the critic
                d_loss_real = self.critic.train_on_batch(imgs, valid)
                d_loss_fake = self.critic.train_on_batch(gen_imgs, fake)
                d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
                # print("piche", d_loss_real, d_loss_fake)

                # Clip critic weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)

            # ---------------------
            #  Train Generator
            # ---------------------

            g_loss = self.combined.train_on_batch(noise, valid)

        # Plot the progress
        # print("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

    def generate_contexts(self, n_samples):
        contexts = np.array([self.generator.predict(np.random.normal(0, 1, (1, self.n_noise)))
                             for k in range(n_samples)])
        return np.round(contexts)


if __name__ == '__main__':
    wgan = WGANMushroom()
    wgan.train(epochs=4000, batch_size=32, sample_interval=50)
    print(wgan.generate_contexts(10))
