# %matplotlib inline

import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random

from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import backend as K

TRAIN_TEST_RATIO = 0.8 # Splits the training and test data 80/20.
learning_rate = 1e-4
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

    con4 = Dense(150, activation='relu')(in_con4)
    cc6 = Dense(150, activation='relu')(in_cc6)

with tf.name_scope("middle_network"):
    #middle = Sequential([
    #    Dense(100, activation='relu', input_shape=(150,)),
    #    Dense(100, activation='relu'),
    #    Dense(100, activation='relu')
    #])
    middle = Dense(100, activation='relu')
    con4_mid = middle(con4)
    cc6_mid = middle(cc6)


with tf.name_scope("output_network"):
    con4_role0 = Dense(50, activation='relu')(con4_mid)
    con4_role0 = Dense(8, activation='softmax')(con4_role0)
    con4_role0_model = Model(inputs=in_con4, outputs=con4_role0)

with tf.name_scope("output_network"):
    con4_role1 = Dense(50, activation='relu')(con4_mid)
    con4_role1 = Dense(8, activation='softmax')(con4_role1)
    con4_role1_model = Model(inputs=in_con4, outputs=con4_role1)

with tf.name_scope("output_network"):
    cc6_role0 = Dense(100, activation='relu')(cc6_mid)
    cc6_role0 = Dense(90, activation='softmax')(cc6_role0)
    cc6_role0_model = Model(inputs=in_cc6, outputs=cc6_role0)

with tf.name_scope("output_network"):
    cc6_role1 = Dense(100, activation='relu')(cc6_mid)
    cc6_role1 = Dense(90, activation='softmax')(cc6_role1)
    cc6_role1_model = Model(inputs=in_cc6, outputs=cc6_role1)

with tf.name_scope("output_network"):
    cc6_role2 = Dense(100, activation='relu')(cc6_mid)
    cc6_role2 = Dense(90, activation='softmax')(cc6_role2)
    cc6_role2_model = Model(inputs=in_cc6, outputs=cc6_role2)

with tf.name_scope("output_network"):
    cc6_role3 = Dense(100, activation='relu')(cc6_mid)
    cc6_role3 = Dense(90, activation='softmax')(cc6_role3)
    cc6_role3_model = Model(inputs=in_cc6, outputs=cc6_role3)

with tf.name_scope("output_network"):
    cc6_role4 = Dense(100, activation='relu')(cc6_mid)
    cc6_role4 = Dense(90, activation='softmax')(cc6_role4)
    cc6_role4_model = Model(inputs=in_cc6, outputs=cc6_role4)

with tf.name_scope("output_network"):
    cc6_role5 = Dense(100, activation='relu')(cc6_mid)
    cc6_role5 = Dense(90, activation='softmax')(cc6_role5)
    cc6_role5_model = Model(inputs=in_cc6, outputs=cc6_role5)


# This creates a model that includes
# the Input layer and three Dense layers
models = [
    (con4_role0_model, in_con4, "connect4", 0, 135, 8),
    (con4_role1_model, in_con4, "connect4", 1, 135, 8),
    (cc6_role0_model, in_cc6,  "chinese_checkers_6", 0, 253, 90),
    (cc6_role1_model, in_cc6,  "chinese_checkers_6", 1, 253, 90),
    (cc6_role2_model, in_cc6,  "chinese_checkers_6", 2, 253, 90),
    (cc6_role3_model, in_cc6,  "chinese_checkers_6", 3, 253, 90),
    (cc6_role4_model, in_cc6,  "chinese_checkers_6", 4, 253, 90),
    (cc6_role5_model, in_cc6,  "chinese_checkers_6", 5, 253, 90),
]

# -------------------------------------------------------------------
#                         Setup Training                             
# -------------------------------------------------------------------

session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())


model_trainigs = []
for model, input_tensor, name, role, state_size, num_actions in models:
    with tf.name_scope("Optimizer"):
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])
                  
    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    train_data, test_data = read_data(file_path, state_size, num_actions, TRAIN_TEST_RATIO)
    print('{0} | train {1} test {2}'.format(file_path, len(train_data),len(test_data)))

    summary_writer = tf.summary.FileWriter('/multygame_keras/{0}/role_{1}'.format(name,role))
    summary_writer.add_graph(session.graph)
    
    train_info = (model, train_data, test_data, summary_writer, name, role)
    model_trainigs.append(train_info)


while True:
    for model, train_data, test_data, summary_writer,name,role in model_trainigs:
        
        train_input = [x[0] for x in train_data]
        train_labels = [x[1] for x in train_data]
        
        test_input = [x[0] for x in test_data]
        test_labels = [x[1] for x in test_data]

        print('Fit: {0} role{1}'.format(name,role))
        model.fit(
            np.array(train_input),
            np.array(train_labels),
            batch_size=64,
            epochs=10,
            validation_data=(np.array(test_input),np.array(test_labels)))




