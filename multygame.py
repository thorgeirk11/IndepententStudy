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

    con4 = Dense(150, activation='relu')(in_con4)
    con4 = Dense(75, activation='relu')(con4)
    
    cc6 = Dense(200, activation='relu')(in_cc6)
    cc6 = Dense(75, activation='relu')(cc6)

    bt = Dense(150, activation='relu')(in_bt)
    bt = Dense(75, activation='relu')(bt)

with tf.name_scope("middle_network"):
    middle = Sequential([
        Dense(200, activation='relu', input_shape=(75,)),
        Dropout(0.5),
        Dense(200, activation='relu')
    ])
    #middle = Dense(100, activation='relu')
    con4_mid = middle(con4)
    cc6_mid = middle(cc6)
    bt_mid = middle(bt)

with tf.name_scope("output_network"):
    bt_role0 = Dense(155, activation='softmax')(bt_mid)
    bt_role0_model = Model(inputs=in_bt, outputs=bt_role0)

with tf.name_scope("output_network"):
    bt_role1 = Dense(155, activation='softmax')(bt_mid)
    bt_role1_model = Model(inputs=in_bt, outputs=bt_role1)

with tf.name_scope("output_network"):
    con4_role0 = Dense(50, activation='relu')(con4_mid)
    con4_role0 = Dense(8, activation='softmax')(con4_role0)
    con4_role0_model = Model(inputs=in_con4, outputs=con4_role0)

with tf.name_scope("output_network"):
    con4_role1 = Dense(50, activation='relu')(con4_mid)
    con4_role1 = Dense(8, activation='softmax')(con4_role1)
    con4_role1_model = Model(inputs=in_con4, outputs=con4_role1)

with tf.name_scope("output_network"):
    #cc6_role0 = Dense(100, activation='relu')(cc6_mid)
    cc6_role0 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role0_model = Model(inputs=in_cc6, outputs=cc6_role0)

with tf.name_scope("output_network"):
    #cc6_role1 = Dense(100, activation='relu')(cc6_mid)
    cc6_role1 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role1_model = Model(inputs=in_cc6, outputs=cc6_role1)

with tf.name_scope("output_network"):
    #cc6_role2 = Dense(100, activation='relu')(cc6_mid)
    cc6_role2 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role2_model = Model(inputs=in_cc6, outputs=cc6_role2)

with tf.name_scope("output_network"):
    #cc6_role3 = Dense(100, activation='relu')(cc6_mid)
    cc6_role3 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role3_model = Model(inputs=in_cc6, outputs=cc6_role3)

with tf.name_scope("output_network"):
    #cc6_role4 = Dense(100, activation='relu')(cc6_mid)
    cc6_role4 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role4_model = Model(inputs=in_cc6, outputs=cc6_role4)

with tf.name_scope("output_network"):
    #cc6_role5 = Dense(100, activation='relu')(cc6_mid)
    cc6_role5 = Dense(90, activation='softmax')(cc6_mid)
    cc6_role5_model = Model(inputs=in_cc6, outputs=cc6_role5)


# This creates a model that includes
# the Input layer and three Dense layers
models = [
    (bt_role0_model, in_bt, "breakthrough", 0, 130, 155),
    (bt_role1_model, in_bt, "breakthrough", 1, 130, 155),

    #(con4_role0_model, in_con4, "connect4", 0, 135, 8),
    #(con4_role1_model, in_con4, "connect4", 1, 135, 8),

    #(cc6_role0_model, in_cc6,  "chinese_checkers_6", 0, 253, 90),
    #(cc6_role1_model, in_cc6,  "chinese_checkers_6", 1, 253, 90),
    #(cc6_role2_model, in_cc6,  "chinese_checkers_6", 2, 253, 90),
    #(cc6_role3_model, in_cc6,  "chinese_checkers_6", 3, 253, 90),
    #(cc6_role4_model, in_cc6,  "chinese_checkers_6", 4, 253, 90),
    #(cc6_role5_model, in_cc6,  "chinese_checkers_6", 5, 253, 90),
]

# -------------------------------------------------------------------
#                         Setup Training                             
# -------------------------------------------------------------------

session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())


train_infos = []
for model, input_tensor, name, role, state_size, num_actions in models:
    with tf.name_scope("Optimizer"):
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])
                  
    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    train_data, test_data = read_data(file_path, state_size, num_actions, TRAIN_TEST_RATIO)
    print('{0} | train {1} test {2}'.format(file_path, len(train_data),len(test_data)))

    tensorboard_callback = TensorBoard(log_dir='/multygame_keras/{0}/role_{1}'.format(name,role))

    train_input = [x[0] for x in train_data]
    train_labels = [x[1] for x in train_data]
    test_input = [x[0] for x in test_data]
    test_labels = [x[1] for x in test_data]

    train_info = (
        model, 
        (train_input, train_labels), 
        (test_input, test_labels), 
        tensorboard_callback
    )
    train_infos.append(train_info)


def optimize(train_infos, epochs):
    for model, train_data, test_data, tensorboard_callback in train_infos:
        model.fit(
            np.array(train_data[0]),
            np.array(train_data[1]),
            batch_size=32,
            epochs=epochs,
            validation_data=(np.array(test_data[0]), np.array(test_data[1])),
            callbacks=[tensorboard_callback]
        )



optimize(train_infos,1000)
