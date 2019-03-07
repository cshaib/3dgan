from __future__ import print_function, division
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Flatten, Dropout, Add, Reshape
from keras.layers import BatchNormalization, Activation, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv3D, Conv1D, UpSampling1D
from keras.models import Sequential, Model, load_model
from keras.optimizers import RMSprop, Adam
from tqdm import tqdm
import keras.backend as K
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

# ========================= 3D GAN and VAE GAN Models ======================= #
class GAN():
    def __init__(self, learning_rate, noise_dim, in_shape=(128, 128, 128, 1), out_shape=(64, 64, 64)):
        # basic attributes
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.optimizer = Adam(0.0002, 0.5)

        # random noise input for the generator
        z = Input(shape=(out_shape))

        # create the models, freeze the discriminator
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()
        self.discriminator.trainable = False

        # create fake image and feed to the discriminator
        img = self.generator(z)
        validity = self.discriminator(img)

        # create stacked model of generator and discriminator
        self.stacked = Model(z, validity)

        # compile models
        # TODO: fix the losses
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        self.stacked.compile(loss='binary_crossentropy', optimizer=optimizer)

    def make_generator(self):
        # architecture: 5x FC_ReLU with {64, 128, 512, 1024, 2048*3} neurons
        out = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        model = Sequential()

        # TODO: Add dimensions as second arguments to each layer
        model.add(Conv3D(filters=512, kernel_size=4, strides=1, activation='relu',
                         input_shape=self.in_shape, padding='same'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=256, kernel_size=4, strides=2,
                  activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=128, kernel_size=4, strides=2,
                  activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=64, kernel_size=4, strides=2,
                  activation='relu', padding='same'))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=1, kernel_size=4, strides=2,
                  activation='relu'))
        model.add(BatchNormalization())
        # self.width * self.height * self.channels
        model.add(Flatten())
        model.add(Dense(out, activation='sigmoid'))
        model.add(Reshape(self.out_shape))
        print(model.summary())

        # TODO: check this
        noise = Input(shape=self.in_shape)
        img = model(noise)

        return Model(noise, img)

    def make_discriminator(self):
        model = Sequential()
        model.add(Conv3D(filters=64, kernel_size=4, strides=2,
                  activation='relu', padding='same', input_shape=self.out_shape))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=512, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization())
        model.add(Conv3D(filters=1, kernel_size=4, strides=1))
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        print(model.summary())

        # TODO: check this
        img = Input(shape=self.out_shape)
        validity = model(img)

        return Model(img, validity)

    def generator_noise(self, n_samples, ndims, mu=0, sigma=0.2):
        # note: in paper is was 0 mean, 0.2 sigma
        return np.random.normal(mu, sigma, (n_samples, ndims))

    def train(self, x_train, epochs, batch, save):

        for e in range(epochs):
            #--------- train discriminator ---------#

            np.random.shuffle(x_train)
            imgs = x_train[:batch//2]

            # TODO: check if out_shape is correct
            noise = self.generator_noise(0, 1, (batch//2, self.out_shape))
            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((batch//2, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros(batch//2, 1))
            d_loss = self.discriminator_loss(d_loss_real, d_loss_fake)

            #--------- train generator ---------#
            noise = self.generator_noise(0, 1, (batch//2, self.out_shape))
            validity_y = np.array([1]*batch)

            g_loss = self.combined.train_on_batch(noise, validity_y)

            print("%d [Discriminator loss: %f, acc.: %.2f%%] [Generator loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
            # TODO:
            # if epoch % save == 0:
            #     self.save(epoch)

    def discriminator_loss(self, d_loss_real, d_loss_fake):
        # TODO: check this is correct, in the paper log D(xi) + log(1 − D(G(zt))).
        return 0.5 * np.add(d_loss_real, d_loss_fake)

    def save(self, e):
        #TODO: save the 3D images at specified intervals
        pass

class VAEGAN(GAN):

    def __init__(self):
        super().__init__()
    # TODO
    def encoder(self):
        pass

    def loss(self):
        pass
