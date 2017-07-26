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
    print(file_name)
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
#                           Create Model                             
# -------------------------------------------------------------------

in_net = Input(shape=(127,))
net = Dense(128, activation='relu')(in_net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(8, activation='softmax')(net)
con4 = [(Model(inputs=in_net, outputs=net), "connect4", 0)]


in_net = Input(shape=(253,))
net = Dense(128, activation='relu')(in_net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(90, activation='softmax')(net)
cc6 = [(Model(inputs=in_net, outputs=net), "chinese_checkers_6", 0)]


in_net = Input(shape=(130,))
net = Dense(128, activation='relu')(in_net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(64, activation='relu')(net)
net = Dense(128, activation='relu')(net)
net = Dense(155, activation='softmax')(net)
bt = [(Model(inputs=in_net, outputs=net), "breakthrough", 0)]



# -------------------------------------------------------------------
#                         Setup Training                             
# -------------------------------------------------------------------


session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())

save_path = '{0}/{1}/saved_weights/{2}/role_{3}.h5'
def load_model(game_info, itteration):
    model, name, role = game_info
    model.load_weights(save_path.format(dir_path, name, itteration, role))
def save_model(game_info, itteration):
    model, name, role = game_info
    model.save_weights(save_path.format(dir_path, name, itteration, role))

def setup_training(game_info):
    model, name, role = game_info
    with tf.name_scope("Optimizer"):
        model.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['acc'])
        # Save Initial weights
        save_model(game_info, 'init')

    file_path = "{0}/data/{0}_role{1}.csv".format(name,role)
    state_size = int(model.get_input_shape_at(0)[1])
    num_actions = int(model.get_output_shape_at(0)[1])
    inputs, labels = read_data(file_path, state_size, num_actions)
    print('{0} | data {1}'.format(file_path, len(inputs)))

    return (
        model, 
        np.array(inputs), 
        np.array(labels), 
        '{0}/multygame_keras/{1}/role_{2}'.format(dir_path, name + "/{0}", role)
    )



def optimize_manual(train_infos, batch_count, pretrained, validation_split, itteration):
    [(model, inputs, labels, log_dir)] = train_infos        
    writer = tf.summary.FileWriter(log_dir.format(itteration) + ('_pretrained' if pretrained else ''))

    batch_size = 128
    total_size = len(inputs) - (len(inputs) % batch_size)

    batch_num = int(total_size / batch_size)
    input_batches = np.split(inputs[:total_size], batch_num)
    label_batches = np.split(labels[:total_size], batch_num)

    batch_num = int(batch_num * validation_split)
    train_input_batches = input_batches[batch_num:]
    train_label_batches = label_batches[batch_num:]

    test_val_input = np.concatenate(input_batches[:batch_num])
    test_val_label = np.concatenate(label_batches[:batch_num])

    test_train_input = np.concatenate(input_batches[batch_num:])
    test_train_label = np.concatenate(label_batches[batch_num:])
    
    epoch = 0
    while epoch < 80000:

        train_loss, train_acc = model.evaluate(test_train_input, test_train_label, batch_size=2000)
        val_loss, val_acc = model.evaluate(test_val_input, test_val_label, batch_size=2000)

        def add_summary(val, tag):
            summary = tf.Summary()
            summary_value = summary.value.add()
            summary_value.simple_value = val
            summary_value.tag = tag
            writer.add_summary(summary, epoch)

        add_summary(train_loss, "train_loss")
        add_summary(train_acc, "train_accuracy")
        add_summary(val_loss, "val_loss")
        add_summary(val_acc, "val_accuracy")
        msg = "{0:>6}: Train Loss {1:>6.1%}  Train Correct {2:>6.1%} | Test Loss {3:>6.1%}  Test Correct {4:>6.1%}"
        print(msg.format(epoch,train_loss,train_acc,val_loss,val_acc))
        writer.flush()

        
        for index in range(batch_num):
            model.train_on_batch(train_input_batches[index], train_label_batches[index])
         
        epoch += batch_num

training = [setup_training(x) for x in con4]
optimize_manual(training, 1000, False, 0.4, 0)
training = [setup_training(x) for x in cc6]
optimize_manual(training, 1000, False, 0.4, 0)
training = [setup_training(x) for x in bt]
optimize_manual(training, 1000, False, 0.4, 0)
