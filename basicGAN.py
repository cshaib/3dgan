from __future__ import print_function, division
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Flatten, Dropout, Add
from keras.layers import BatchNormalization, Activation, MaxPooling1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv1D, UpSampling1D
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
class basicGAN():

    def __init__(self, learning_rate, noise_dim, in_shape, out_shape):
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.in_shape = in_shape
        self.out_shape = out_shape

    def generator():
        # In/Out: Depends on data and preprocessing
        # architecture: 5x FC_ReLU with {64, 128, 512, 1024, 2048*3} neurons

        #inputs = Input(shape=self.shape)
        model = Sequential()

        # TODO: Add dimensions as second arguments to each layer
        model.add(Dense(64,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(128,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512,))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1024,))
        model.add(LeakyReLU(alpha=0.2))
        # self.width * self.height * self.channels
        model.add(Dense(2048*3, activation = 'sigmoid'))

        model.add(reshape((self.width, self.height, self.channels)))

        return model

    def discriminator():
        model = Sequential()

        model.add(Conv1D(filters=64, kernel_size=1,stride=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=128, kernel_size=1,stride=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=256, kernel_size=1,stride=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=256, kernel_size=1,stride=1))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(filters=512, kernel_size=1,stride=1))
        model.add(LeakyReLU(alpha=0.2))

        #TODO: add num_features here.. they do feature wise pooling
        model.add(MaxPooling1D(pool_size=num_features))

        model.add(Dense(128, ))
        model.add(Dense(64, ))
        model.add(Dense(1, activation='sigmoid'))

        model.summary()
        return model

    def generator_noise(self, n_samples, ndims, mu=0, sigma=0.2):
        # note: in paper is was 0 mean, 0.2 sigma
        return np.random.normal(mu, sigma, (n_samples, ndims))

    # TODO: define loss (adversarial and generator)
    # TODO: