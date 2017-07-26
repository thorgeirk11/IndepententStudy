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
#                           Create Model                             
# -------------------------------------------------------------------

with tf.name_scope("input_connect4"):
    in_con4 = Input(shape=(127,))
    con4 = Dense(200, activation='relu')(in_con4)

with tf.name_scope("input_network_chinses_checkers_6"):
    in_cc6 = Input(shape=(253,))
    cc6 = Dense(200, activation='relu')(in_cc6)

with tf.name_scope("input_breakthrough"):
    in_bt = Input(shape=(130,))
    bt = Dense(200, activation='relu')(in_bt)

with tf.name_scope("middle_layers"):
    middle = Sequential([
        Dense(500, activation='relu', input_shape=(200,)),
        Dropout(0.5),
        Dense(500, activation='relu')
    ])
    con4_mid = middle(con4)
    cc6_mid = middle(cc6)
    bt_mid = middle(bt)

con4_models = []
for i in range(2):
    with tf.name_scope("output_conncet4_role{0}".format(i)):
        out = Dense(50, activation='relu')(con4_mid)
        out = Dense(8, activation='softmax')(out)
        model = (Model(inputs=in_con4, outputs=out), "connect4", i)
        con4_models.append(model)

cc6_models = []
for i in range(6):
    with tf.name_scope("output_chinese_checkers_role{0}".format(i)):
        out = Dense(200, activation='relu')(cc6_mid)
        out = Dense(90, activation='softmax')(out)
        model = (Model(inputs=in_cc6, outputs=out), "chinese_checkers_6", i)
        cc6_models.append(model)

bt_models = []
for i in range(2):
    with tf.name_scope("output_breakthrough_role{0}".format(i)):
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

def print_eval(i):
    eval =      model.evaluate(inputs, labels, verbose=0)
    eval_test = model.evaluate(inputs_test, labels_test, verbose=0)
    msg = "{0:>6}: Train Loss {1:>6.1%}  Train Correct {2:>6.1%} | Test Loss {3:>6.1%}  Test Correct {4:>6.1%}"
    labels_pred = model.predict(inputs_test)


with tf.name_scope("Train"):
    def optimize(train_infos, epochs, use_tensorboard, pretrained, validation_split, itteration):
        for model, inputs, labels, log_dir in train_infos:
            callbacks = []
            if use_tensorboard:
                callbacks.append(
                    TensorBoard(
                        log_dir= log_dir + ('_pretrained' if pretrained else ''), #+ '_' + str(itteration),
                        histogram_freq=5,
                        write_grads=True
                    )
                )

            model.fit(
                inputs,
                labels,
                batch_size=5000,
                epochs=epochs,
                validation_split=validation_split,
                callbacks=callbacks
            )

    def optimize_manual(train_infos, batch_count, use_tensorboard, pretrained, validation_split, itteration):
        for model, inputs, labels, log_dir in train_infos:            
            writer = tf.summary.FileWriter(log_dir.format(itteration) + ('_pretrained' if pretrained else ''))

            batch_size = 128
            total_size = len(inputs) - (len(inputs) % batch_size)

            batch_num = int(total_size / batch_size)
            input_batches = np.split(inputs[:total_size], batch_num)
            label_batches = np.split(labels[:total_size], batch_num)

            batch_num = int(batch_num * validation_split)
            train_input_batches = input_batches[batch_num:]
            train_label_batches = label_batches[batch_num:]

            test_input = np.concatenate(input_batches[:batch_num])
            test_label = np.concatenate(label_batches[:batch_num])

            for i in range(batch_count):
                index = i % batch_num
                
                train_loss, train_acc = model.train_on_batch(train_input_batches[index], train_label_batches[index])
                if i % 20 == 0:
                    val_loss, val_acc = model.evaluate(test_input, test_label, batch_size=2000)
                    print(" ----- {0} / {1} ".format(i, batch_count))

                    def add_summary(val, tag):
                        summary = tf.Summary()
                        summary_value = summary.value.add()
                        summary_value.simple_value = val
                        summary_value.tag = tag
                        writer.add_summary(summary, i)

                    add_summary(train_loss, "train_loss")
                    add_summary(train_acc, "train_accuracy")
                    add_summary(val_loss, "val_loss")
                    add_summary(val_acc, "val_accuracy")

                    writer.flush()

def run(train_models, models, itteration):
    for model, _, _ in models:
        weights = [np.random.permutation(w) for w in model.get_weights()]
        model.set_weights(weights)

    optimize_manual(train_models[:1], 3000, True, False, 0.4, itteration)

    for model in models:
        load_model(model, 'init')

    if len(train_models) > 2:
        for i in range(500):
            optimize(train_models[1:], 1, False, False, 0, itteration)
    else:
        optimize(train_models[1:], 500, False, False, 0, itteration)

    optimize_manual(train_models[:1], 3000, True, True, 0.4, itteration)
    

bt_training =   [setup_training(x) for x in bt_models]
con4_training = [setup_training(x) for x in con4_models]
cc6_training =  [setup_training(x) for x in cc6_models]

itteration = 1
while True:
    run(bt_training, bt_models, itteration)
    run(con4_training, con4_models, itteration)
    run(cc6_training, cc6_models, itteration)
    itteration += 1
