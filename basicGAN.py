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

# =============================== Load Dataset ============================= #

# Depending on what the data looks like, length/width/height should be adjusted
# Assuming all augmentation and pre-processing steps are done
# Training and testing splits remain the same

# length =
# width =
# height ='

# Normalize the data (not sure if necessary?)
# Split into training, validation, and test sets

# ========================= Models and Loss Functions ======================= #

class GAN():
    def __init__(self, learning_rate, noise_dim, in_shape=(128, 128, 128, 1), out_shape=(64, 64, 64)):
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.generator = self.make_generator()
        self.discriminator = self.make_discriminator()

    def make_generator(self):
        # In/Out: Depends on data and preprocessing
        # architecture: 5x FC_ReLU with {64, 128, 512, 1024, 2048*3} neurons

        out = self.out_shape[0] * self.out_shape[1] * self.out_shape[2]
        # inputs = Input(shape=self.shape)
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

    def discriminator_loss(self, d_loss_real, d_loss_fake):
        # d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        # d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # in the paper log D(xi) + log(1 − D(G(zt))).
        return 0.5 * np.add(d_loss_real, d_loss_fake)

    def generator_loss(self):
        # in the paper log(1 − D(G(zt))) + ||G(E(yi)) − xi||2
        return

class VAEGAN(GAN):

    def __init__(self):
        super().__init__()
    # TODO
    def encoder(self):
        pass

    def loss(self):
        pass
