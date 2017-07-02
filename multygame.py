# %matplotlib inline

import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random

from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import backend as K
from multy_gpu import make_parallel
import os

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

VALIDATION_SPLIT = 0.8 # Splits the training and  data 20/80.
metadata_size = 4 
GPU_COUNT = len(get_available_gpus())

dir_path = os.getcwd()

# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, state_size, num_actions):
    meta = pd.read_csv(file_name, usecols = range(0,metadata_size), header = None)
    features = pd.read_csv(file_name, usecols = range(metadata_size, metadata_size + state_size), header = None)
    labels = pd.read_csv(file_name, usecols = range(metadata_size + state_size, metadata_size + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if item[0][3] != 0]
    inputs = [x[0] for x in data]
    labels = [x[1] for x in data]
    return inputs, labels

# -------------------------------------------------------------------
#                           Create Model                             
# -------------------------------------------------------------------

with tf.name_scope("input_network"):
    in_con4 = Input(shape=(135,))
    in_cc6 = Input(shape=(253,))
    in_bt = Input(shape=(130,))

    con4 = Dense(200, activation='relu')(in_con4)
    #con4 = Dense(75, activation='relu')(con4)
    
    cc6 = Dense(200, activation='relu')(in_cc6)
    #cc6 = Dense(75, activation='relu')(cc6)

    bt = Dense(200, activation='relu')(in_bt)
    #bt = Dense(75, activation='relu')(bt)

with tf.name_scope("middle_network"):
    middle = Sequential([
        Dense(500, activation='relu', input_shape=(200,)),
        Dropout(0.5),
        Dense(500, activation='relu')
    ])
    #middle = Dense(100, activation='relu')
    con4_mid = middle(con4)
    cc6_mid = middle(cc6)
    bt_mid = middle(bt)



con4_models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(50, activation='relu')(con4_mid)
        out = Dense(8, activation='softmax')(out)
        model = (Model(inputs=in_con4, outputs=out), "connect4", i)
        con4_models.append(model)

cc6_models = []
for i in range(6):
    with tf.name_scope("output_network"):
        out = Dense(200, activation='relu')(cc6_mid)
        out = Dense(90, activation='softmax')(out)
        model = (Model(inputs=in_cc6, outputs=out), "chinese_checkers_6", i)
        cc6_models.append(model)

bt_models = []
for i in range(2):
    with tf.name_scope("output_network"):
        out = Dense(200, activation='relu')(bt_mid)
        out = Dense(155, activation='softmax')(out)
        model = (Model(inputs=in_bt, outputs=out), "breakthrough", i)
        bt_models.append(model)

all_game_models = bt_models + con4_models + cc6_models 

# -------------------------------------------------------------------
#                         Setup Training                             
# -------------------------------------------------------------------


session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())

def reset_training(game_info):
    model, name, role = game_info
    model.load_weights('{0}/{1}/saved_weights/init/role_{2}.h5'.format(dir_path, name, role))

def setup_training(game_info):
    model, name, role = game_info
    with tf.name_scope("Optimizer"):
        if GPU_COUNT > 1:
            make_parallel(model, GPU_COUNT)
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])
        # Save Initial weights
        model.save_weights('{0}/{1}/saved_weights/init/role_{2}.h5'.format(dir_path, name, role))

    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    state_size = int(model.input.shape[1])
    num_actions = int(model.output.shape[1])
    inputs, labels = read_data(file_path, state_size, num_actions)
    print('{0} | data {1}'.format(file_path, len(inputs)))

    return (
        model, 
        inputs, 
        labels, 
        '{0}/multygame_keras/{1}/role_{2}'.format(dir_path, name, role)
    )


with tf.name_scope("Train"):
    def optimize(train_infos, epochs, use_tensorboard, pretrained, validation_split):
        for model, inputs, labels, log_dir in train_infos:
            
            callbacks = []
            if use_tensorboard:
                callbacks.append(
                    TensorBoard(
                        log_dir= log_dir + ('_pretrained' if pretrained else ''),
                        histogram_freq=5,
                        write_grads=True
                    )
                )

            model.fit(
                np.array(inputs),
                np.array(labels),
                batch_size=32 * GPU_COUNT,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks
            )

    bt_training =   [setup_training(x) for x in bt_models]
    con4_training = [setup_training(x) for x in con4_models]
    cc6_training =  [setup_training(x) for x in cc6_models]


for i in range(5):
    optimize(cc6_training[1:], 5, False, False, 0)

optimize(cc6_training[:1], 15, True, True, 0.3)

# Reset the weights
for model_info in cc6_models:
    reset_training(model_info)

optimize(cc6_training[:1], 15, True, False, 0.3)