from tensorflow.keras import Model, layers
from tensorflow.keras.layers import Input, Conv1D, Conv2D, BatchNormalization, \
    Concatenate, Softmax
from tensorflow import math, exp
# from utils import SoftmaxMap
from data_transform import read_transform
import numpy as np


class SoftmaxMap(layers.Layer):
    def __init__(self, axis=-1, **kwargs):
        self.axis = axis
        super(SoftmaxMap, self).__init__(**kwargs)

    def build(self, input_shape):
        pass

    def call(self, x, mask=None):
        e = exp(x - math.reduce_max(x, axis=self.axis, keepdims=True))
        s = math.reduce_sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape_for(self, input_shape):
        return input_shape


def train_model():
    X_train, X_val, X_test, Y_train, Y_val, Y_test = read_transform()
    print(f'shapes: {X_train.shape}, {X_val.shape}, {Y_train.shape}, {Y_val.shape}')
    
    # X_train = np.array(X_train).reshape((-1, 9, 9, 1))
    # X_val = np.array(X_val).reshape((-1, 9, 9, 1))
    # X_test = np.array(X_test).reshape((-1, 9, 9, 1))
    # Y_train = np.array(Y_train).reshape((-1, 9, 9, 1))
    # Y_val = np.array(Y_val).reshape((-1, 9, 9, 1))
    # Y_test = np.array(Y_test).reshape((-1, 9, 9, 1))

    # if len(X_train[0].shape) == 2:
    #     X_train = [item.reshape(9, 9, 1) for item in X_train]
    #     X_val = [item.reshape(9, 9, 1) for item in X_val]
    #     X_test = [item.reshape(9, 9, 1) for item in X_test]
        # X_train = X_train.reshape((-1, 9, 9, 1))
        # X_val = X_val.reshape((-1, 9, 9, 1))
        # X_test = X_test.reshape((-1, 9, 9, 1))
    
    model = build_model()
    # --- compile and fit the model
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    model.fit(X_train, Y_train, epochs=30, batch_size=128,
              validation_data=(X_val, Y_val))

    model.save("../alpha_sudoku/policy_network.keras")


def build_model():
    # Model definition
    # input = Input(shape=(9, 9, 9))
    input = Input(shape=(9, 9, 3))

    print(f'input shape: {input.shape}')
    x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                activation='tanh')(input)
    print(f'x1 shape: {x1.shape}')

    x2 = Conv2D(32, kernel_size=(1, 9), strides=(1, 1), padding='same',
                activation='tanh')(input)
    print(f'x2 shape: {x2.shape}')

    x3 = Conv2D(32, kernel_size=(9, 1), strides=(1, 1), padding='same',
                activation='tanh')(input)
    print(f'x3 shape: {x3.shape}')

    x = Concatenate()([x1, x2, x3])
    print(f'x shape (1): {x.shape}')

    x = BatchNormalization()(x)
    print(f'x shape (2): {x.shape}')

    x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
                activation='tanh')(x)
    print(f'x shape (3): {x.shape}')

    x = BatchNormalization()(x)
    print(f'x shape (4): {x.shape}')

    x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
                activation='relu')(x)
    print(f'x shape (5): {x.shape}')

    x = BatchNormalization()(x)
    print(f'x shape (6): {x.shape}')

    x = Concatenate()([x, input])
    print(f'x shape (7): {x.shape}')

    x = Conv2D(3, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)
    print(f'x shape (8): {x.shape}')

    # x1 = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #             activation='tanh')(input)
    # x2 = Conv2D(32, kernel_size=(1, 9), strides=(1, 1), padding='same',
    #             activation='tanh')(input)
    # x3 = Conv2D(32, kernel_size=(9, 1), strides=(1, 1), padding='same',
    #             activation='tanh')(input)
    # x4 = Conv2D(32, kernel_size=(9, 9), strides=(1, 1), padding='same',
    #             activation='tanh')(input)
    # x = Concatenate()([x1, x2, x3, x4])
    # x = BatchNormalization()(x)
    # x = Conv2D(64, kernel_size=(9, 9), strides=(1, 1), padding='same',
    #            activation='tanh')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(64, kernel_size=(7, 7), strides=(1, 1), padding='same',
    #            activation='tanh')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(64, kernel_size=(5, 5), strides=(1, 1), padding='same',
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same',
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #            activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same',
    #            activation='relu')(x)
    # x = BatchNormalization()(x)

    # x = Concatenate()([x, input])
    # x = Conv2D(9, kernel_size=(1, 1), strides=(1, 1), padding='same')(x)

    outputs_play = Softmax()(x)

    # outputs_play = SoftmaxMap()(x)

    # Model instantiation
    model = Model(input, outputs_play)
    # print(model.summary())

    return model


if __name__ == '__main__':
    train_model()