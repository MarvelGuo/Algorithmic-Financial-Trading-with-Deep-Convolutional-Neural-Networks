"""
Created on 2020/4/15 17:37
Author: Xinyu Guo
Email: xyguo@bu.edu
IDE: PyCharm
"""
from Model_Analysis import *
from warnings import simplefilter
simplefilter(action='ignore', category=Warning)
import time

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, LeakyReLU
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, Callback
from tensorflow.keras.utils import get_custom_objects


def CNN_construct(params, ticker, best_model_path=None):
    if best_model_path is None:
        best_model_path = '.\\Data\\' + ticker+ '\\best_model\\model' + time.strftime("%Y%m%d_%H%M", time.localtime())

    model = Sequential()
    conv_layer1 = Conv2D(filters=params["conv2d_layers1"]["filters"],
                         kernel_size=params["conv2d_layers1"]["kernel_size"],
                         strides=params["conv2d_layers1"]["strides"],
                         kernel_regularizer=regularizers.l2(
                             params["conv2d_layers1"]["kernel_regularizer"]),
                         padding=params["conv2d_layers1"]["padding"], activation="relu", use_bias=True,
                         kernel_initializer=params['conv2d_layers1']['kernel_initializer'],
                         input_shape=(params['input_dim_1'],
                                      params['input_dim_2'], params['input_dim_3']))
    model.add(conv_layer1)
    if params["conv2d_layers1"]['maxpool'] != 0:
        model.add(MaxPool2D(pool_size=params["conv2d_layers1"]['maxpool']))
    model.add(Dropout(params['conv2d_layers1']['dropout']))

    conv_layer2 = Conv2D(filters=params["conv2d_layers2"]["filters"],
                         kernel_size=params["conv2d_layers2"]["kernel_size"],
                         strides=params["conv2d_layers2"]["strides"],
                         kernel_regularizer=regularizers.l2(
                             params["conv2d_layers2"]["kernel_regularizer"]),
                         padding=params["conv2d_layers2"]["padding"], activation="relu", use_bias=True,
                         kernel_initializer=params['conv2d_layers2']['kernel_initializer'])
    model.add(conv_layer2)
    model.add(MaxPool2D(pool_size=params["conv2d_layers2"]['maxpool']))
    model.add(Dropout(params['conv2d_layers2']['dropout']))

    model.add(Flatten())
    model.add(Dense(params['dense_layers']["dense_nodes_1"],
                    activation='relu'))  # FC layer
    model.add(Dropout(params['dense_layers']["dense_do_1"]))

    model.add(Dense(3, activation='softmax'))

    ##########################################
    optimizer = optimizers.Adam(learning_rate=params["lr"], beta_1=0.9, beta_2=0.999, amsgrad=False)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy', f1_metric])

    get_custom_objects().update({"f1_metric": f1_metric})
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=100, min_delta=0.0001)
    rlp = ReduceLROnPlateau(monitor='val_loss', factor=0.02, patience=10, verbose=1, mode='min',
                            min_delta=0.001, cooldown=1, min_lr=0.0001)
    mcp = ModelCheckpoint(best_model_path, verbose=0,
                          save_best_only=True, save_weights_only=False, mode='min', save_freq='epoch')

    return model, es, rlp, mcp


