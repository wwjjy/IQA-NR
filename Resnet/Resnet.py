from keras.layers import merge
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D, AveragePooling2D
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.layers import Input
from keras.optimizers import adam
import os


def identity_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter
    out = Convolution2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out


def conv_block(x, nb_filter, kernel_size=3):
    k1, k2, k3 = nb_filter

    out = Convolution2D(k1, 1, 1)(x)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k2, kernel_size, kernel_size, border_mode='same')(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)

    out = Convolution2D(k3, 1, 1)(out)
    out = BatchNormalization()(out)

    x = Convolution2D(k3, 1, 1)(x)
    x = BatchNormalization()(x)

    out = merge([out, x], mode='sum')
    out = Activation('relu')(out)
    return out


class ResNet:
    def __init__(self, weights_path=None):
        self.inp = Input(shape=(224, 224, 1))
        self.out = ZeroPadding2D((3, 3))(self.inp)
        self.out = Convolution2D(64, 7, 7, subsample=(2, 2))(self.out)
        self.out = BatchNormalization()(self.out)
        self.out = Activation('relu')(self.out)
        self.out = MaxPooling2D((3, 3), strides=(2, 2))(self.out)

        self.out = conv_block(self.out, [64, 64, 256])
        self.out = identity_block(self.out, [64, 64, 256])
        self.out = identity_block(self.out, [64, 64, 256])

        self.out = conv_block(self.out, [128, 128, 512])
        self.out = identity_block(self.out, [128, 128, 512])
        self.out = identity_block(self.out, [128, 128, 512])
        self.out = identity_block(self.out, [128, 128, 512])

        self.out = conv_block(self.out, [256, 256, 1024])
        self.out = identity_block(self.out, [256, 256, 1024])
        self.out = identity_block(self.out, [256, 256, 1024])
        self.out = identity_block(self.out, [256, 256, 1024])
        self.out = identity_block(self.out, [256, 256, 1024])
        self.out = identity_block(self.out, [256, 256, 1024])

        self.out = conv_block(self.out, [512, 512, 2048])
        self.out = identity_block(self.out, [512, 512, 2048])
        self.out = identity_block(self.out, [512, 512, 2048])

        self.out = AveragePooling2D((7, 7))(self.out)
        self.out = Flatten()(self.out)
        self.out = Dense(1)(self.out)

        self.model = Model(self.inp, self.out)
        self.model.compile(optimizer=adam(), loss='mse')

        if os.path.exists(weights_path):
            self.model.load_weights(weights_path)
            print("********************Load Model Success*********************")

    def train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def show_summary(self):
        self.model.summary()

    def save_parameter(self, modelpath):
        self.model.save(modelpath)

    def predict(self, x):
        return self.model.predict(x)
