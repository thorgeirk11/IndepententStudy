# %matplotlib inline

import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random

from keras.callbacks import TensorBoard
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import backend as K

TRAIN_TEST_RATIO = 0.8 # Splits the training and test data 80/20.
metadata_size = 4 


# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, state_size, num_actions, train_data_size):
    meta = pd.read_csv(file_name, usecols = range(0,metadata_size), header = None)
    features = pd.read_csv(file_name, usecols = range(metadata_size, metadata_size + state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(metadata_size + state_size, metadata_size + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if item[0][3] != 0]

    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data =  data[cut:]
    return train_data, test_data

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
        Dense(1000, activation='relu', input_shape=(200,)),
        Dropout(0.5),
        Dense(1000, activation='relu')
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


def setup_training(game_info):
    model, name, role = game_info
    with tf.name_scope("Optimizer"):
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])
    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    state_size = int(model.input.shape[1])
    num_actions = int(model.output.shape[1])
    train_data, test_data = read_data(file_path, state_size, num_actions, TRAIN_TEST_RATIO)
    print('{0} | train {1} test {2}'.format(file_path, len(train_data),len(test_data)))


    train_input = [x[0] for x in train_data]
    train_labels = [x[1] for x in train_data]
    test_input = [x[0] for x in test_data]
    test_labels = [x[1] for x in test_data]
    
    return (
        model, 
        (train_input, train_labels), 
        (test_input, test_labels), 
        '/multygame_keras/{0}/role_{1}'.format(name,role)
    )


with tf.name_scope("Train"):
    def optimize(train_infos, epochs, use_tensorboard, pretrained):
        for model, train_data, test_data, log_dir in train_infos:
            
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
                np.array(train_data[0]),
                np.array(train_data[1]),
                batch_size=32,
                epochs=epochs,
                validation_data=(np.array(test_data[0]), np.array(test_data[1])),
                callbacks=callbacks
            )

    bt_training =   [setup_training(x) for x in bt_models]
    con4_training = [setup_training(x,) for x in con4_models]
    cc6_training =    [setup_training(x) for x in cc6_models]

    for i in range(5):
        optimize(cc6_training[1:], 5, False, False)
    
    optimize(cc6_training[:1], 15, True, True)
