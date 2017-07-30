# %matplotlib inline

import tensorflow as tf
import numpy as np
import numpy
import pandas as pd
import random

from tensorflow.python.client import device_lib
from keras.callbacks import TensorBoard, ReduceLROnPlateau
from keras.layers import Input, Dense, Dropout
from keras.models import Model, Sequential
from keras import backend as K
import os

VALIDATION_SPLIT = 0.8 # Splits the training and test data 20/80.
metadata_size = 4 

dir_path = os.getcwd()

# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, state_size, num_actions):
    meta = pd.read_csv(file_name, usecols = range(0,metadata_size), header = None)
    features = pd.read_csv(file_name, usecols = range(metadata_size, metadata_size + state_size), header = None)
    labels = pd.read_csv(file_name, usecols = range(metadata_size + state_size, metadata_size + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    random.shuffle(data)
    data = [(item[1],item[2]) for item in data if item[0][3] != 0]
    inputs = [x[0] for x in data]
    labels = [x[1] for x in data]
    return inputs, labels


# -------------------------------------------------------------------
#                         Setup Training                             
# -------------------------------------------------------------------


session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())

def setup_training(name, input, output, last_layer):
    
    net_in = Input(shape=(input,))
    net = Dense(200, activation='relu')(net_in)
    net = Dense(500, activation='relu')(net)
    net = Dropout(0.5)(net)
    net = Dense(500, activation='relu')(net)
    net = Dense(last_layer, activation='relu')(net)
    net = Dense(output, activation='softmax')(net)
    model = Model(inputs=net_in, outputs=net)
    
    role = 0
    with tf.name_scope("Optimizer"):
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])

    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    state_size = int(model.get_input_shape_at(0)[1])
    num_actions = int(model.get_output_shape_at(0)[1])
    inputs, labels = read_data(file_path, state_size, num_actions)
    print('{0} | data {1}'.format(file_path, len(inputs)))

    return (
        model, 
        np.array(inputs), 
        np.array(labels), 
        '{0}/single_role_tests/{1}/role_{2}'.format(dir_path, name + "/{0}", role)
    )



def optimize(train_info, validation_split, itteration):
    model, inputs, labels, log_dir = train_info        

    print(log_dir.format(itteration))
    writer = tf.summary.FileWriter(log_dir.format(itteration))

    batch_size = 128
    total_size = len(inputs) - (len(inputs) % batch_size)

    batch_num = int(total_size / batch_size)
    input_batches = np.split(inputs[:total_size], batch_num)
    label_batches = np.split(labels[:total_size], batch_num)

    batch_num = int(batch_num * validation_split)
    train_input_batches = input_batches[batch_num:]
    train_label_batches = label_batches[batch_num:]

    val_input = np.concatenate(input_batches[:batch_num])
    val_label = np.concatenate(label_batches[:batch_num])
    
    epoch = 0
    while epoch < 50000:
        val_loss, val_acc = model.evaluate(val_input, val_label, batch_size=5000)

        def add_summary(val, tag):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = val
            summary_value.tag = tag
            writer.add_summary(summary, epoch)

        add_summary(val_acc, "val_accuracy")
        writer.flush()

        msg = "{0:>6}: Test Loss {1:>6.1%}  Test Correct {2:>6.1%}"
        print(msg.format(epoch, val_loss, val_acc))
        for index in range(batch_num):
            model.train_on_batch(train_input_batches[index], train_label_batches[index])
         
        epoch += batch_num
    writer.close()

for i in range(1,101):
    con4_model = setup_training("connect4", 127, 8, 50)
    optimize(con4_model, 0.4, i)
    del con4_model
    
    cc6_model = setup_training("chinese_checkers_6", 253, 90, 200)
    optimize(cc6_model, 0.4, i)
    del cc6_model

    bt_model = setup_training("breakthrough", 130, 155, 200)
    optimize(bt_model, 0.4, i)
    del bt_model