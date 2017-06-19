# %matplotlib inline

import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random

from keras.layers import Input, Dense, Dropout, concatenate
from keras.models import Model
from keras import backend as K

TRAIN_TEST_RATIO = 0.8 # Splits the training and test data 80/20.
learning_rate = 1e-4


in_con4 = Input(shape=(135,))
in_cc6 = Input(shape=(253,))

con4 = Dense(64, activation='relu')(in_con4)
cc6 = Dense(64, activation='relu')(in_cc6)

middle = concatenate([con4, cc6], axis=-1)
middle = Dense(100, activation='relu')(middle)

# Conncet 4
con4_role0 = Dense(50, activation='relu')(middle)
con4_role0 = Dense(8, activation='softmax')(con4_role0)
con4_role0_model = Model(inputs=in_con4, outputs=con4_role0)

con4_role1 = Dense(50, activation='relu')(middle)
con4_role1 = Dense(8, activation='softmax')(con4_role1)
con4_role1_model = Model(inputs=in_con4, outputs=con4_role1)

cc6_role0 = Dense(100, activation='relu')(middle)
cc6_role0 = Dense(90, activation='softmax')(cc6_role0)
cc6_role0_model = Model(inputs=in_cc6, outputs=cc6_role0)

cc6_role1 = Dense(100, activation='relu')(middle)
cc6_role1 = Dense(90, activation='softmax')(cc6_role1)
cc6_role1_model = Model(inputs=in_cc6, outputs=cc6_role1)

cc6_role2 = Dense(100, activation='relu')(middle)
cc6_role2 = Dense(90, activation='softmax')(cc6_role2)
cc6_role2_model = Model(inputs=in_cc6, outputs=cc6_role2)

cc6_role3 = Dense(100, activation='relu')(middle)
cc6_role3 = Dense(90, activation='softmax')(cc6_role3)
cc6_role3_model = Model(inputs=in_cc6, outputs=cc6_role3)

cc6_role4 = Dense(100, activation='relu')(middle)
cc6_role4 = Dense(90, activation='softmax')(cc6_role4)
cc6_role4_model = Model(inputs=in_cc6, outputs=cc6_role4)

cc6_role5 = Dense(100, activation='relu')(middle)
cc6_role5 = Dense(90, activation='softmax')(cc6_role5)
cc6_role5_model = Model(inputs=in_cc6, outputs=cc6_role5)


# This creates a model that includes
# the Input layer and three Dense layers
models = [
    con4_role0_model, 
    con4_role1_model, 
    cc6_role0_model, 
    cc6_role1_model,
    cc6_role2_model,
    cc6_role3_model,
    cc6_role4_model,
    cc6_role5_model,
]

for model in models:
    model.compile(optimizer='adam',
                  loss='mean_squared_error')


def read_data(file_name, state_size,num_actions, train_data_size):
    meta = pd.read_csv(file_name, usecols = range(0,4), header = None)
    features = pd.read_csv(file_name, usecols = range(4, 4 + state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(4 + state_size, 4 + state_size + num_actions), header = None)
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if item[3] != 0]
    return  features ,labels

#data_a, labels = read_data("Connect4/data/connect4_role{0}.csv", 134, 8, 0.8)

model.summary(100)

#model.fit([data_a, data_b], labels, epochs=10)
