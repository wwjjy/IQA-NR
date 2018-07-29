from keras import backend as k
from keras.layers import merge, Input
from keras.layers.convolutional import Convolution2D, AveragePooling2D
from keras.layers.core import Activation, Flatten, Dense
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
import keras
import numpy as np
import os


def my_loss(y_true, y_pred):
    """
    这玩意是我自己根据论文写得的一个渣渣损失函数,也不知实际效果到底好不好,先这样吧
    :param y_true:  这个是输入的标签，也就是真实值
    :param y_pred:  这个是网络的输出，也就是预测值
    :return:        预测值和真实值之间的误差
    """

    loss = 0
    # 在这里首先应该是要遍历预测值，然后两两配对求loss
    for i in range(k.shape(y_pred)[0]-1):
        for j in range(i+1, k.shape(y_pred)[0]):
            temp_a = y_pred[i]
            temp_b = y_pred[j]
            if y_true[i] > y_true[j]:       # i > j 与默认的i < j 相反， 所以此时这种情况需要变为 j在前 i在后
                loss = loss + np.exp(temp_b - temp_a)
            elif y_true[i] < y_true[j]:     # i < j 与默认的情况相同，直接处理
                loss = loss + np.exp(temp_a - temp_b)
            else:                           # i = j 的情况，暂时先不考虑
                pass

    return loss


# 自己构建了一个渣渣残差块，希望效果好咯
def residual_block(x1, x2, nb_filter, kernel_size=3):
    k1, k2 = nb_filter
    cov1 = Convolution2D(k1, kernel_size, kernel_size, border_mode='same')
    out1 = cov1(x1)
    out2 = cov1(x2)
    out1 = BatchNormalization()(out1)
    out2 = BatchNormalization()(out2)
    out1 = Activation('relu')(out1)
    out2 = Activation('relu')(out2)

    cov2 = Convolution2D(k1, kernel_size, kernel_size, border_mode='same')
    out1 = cov2(out1)
    out2 = cov2(out2)
    out1 = BatchNormalization()(out1)
    out2 = BatchNormalization()(out2)

    out1 = merge([out1, x1], mode='sum')
    out2 = merge([out2, x2], mode='sum')
    out1 = Activation('relu')(out1)
    out2 = Activation('relu')(out2)
    return out1, out2


class ReqNet:
    def __init__(self, h_layers=3, weight_path='haha'):
        self.inp1 = Input(shape=(224, 224, 1))
        self.inp2 = Input(shape=(224, 224, 1))
        self.con1 = Convolution2D(32, 3, 3, border_mode='same')
        self.out1 = self.con1(self.inp1)
        self.out2 = self.con1(self.inp2)
        self.BN1 = BatchNormalization()
        self.out1 = self.BN1(self.out1)
        self.out2 = self.BN1(self.out2)
        self.out1 = Activation('relu')(self.out1)
        self.out2 = Activation('relu')(self.out2)

        for i in range(h_layers):
            self.out1, self.out2 = residual_block(self.out1, self.out2, [32, 32])

        self.out1 = AveragePooling2D((7, 7))(self.out1)
        self.out2 = AveragePooling2D((7, 7))(self.out2)
        self.out1 = Flatten()(self.out1)
        self.out2 = Flatten()(self.out2)

        self.D1 = Dense(200)
        self.out1 = self.D1(self.out1)
        self.feature = self.out1
        self.out2 = self.D1(self.out2)
        self.out1 = Activation('relu')(self.out1)

        self.out2 = Activation('relu')(self.out2)
        self.D2 = Dense(1)
        self.out1 = self.D2(self.out1)
        self.out2 = self.D2(self.out2)
        self.out = keras.layers.Subtract()([self.out1, self.out2])

        self.model = Model(inputs=[self.inp1, self.inp2], outputs=self.out)
        self.model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08), loss='mse')

        if os.path.exists(weight_path):
            self.model.load_weights(weight_path)
            print("********************Load Model Success*********************")

    def train_on_batch(self, x1, x2, y):
        return self.model.train_on_batch([x1, x2], y)

    def show_summary(self):
        self.model.summary()

    def save_parameter(self, modelpath):
        self.model.save(modelpath)

    def predict(self, x1, x2):
        return self.model.predict([x1, x2])
