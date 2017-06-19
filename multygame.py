# %matplotlib inline

import tensorflow as tf
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import numpy
import pandas as pd
import random

from keras.layers import Dense, Dropout
from keras import backend as K

input_formats = [
    # ( file_name ( metadata_size, state_size, num_actions ) num_roles )
    ("Connect4/data/connect4_role{0}.csv",              (4, 135, 8 ), 2), 
    ("ChineseCheckers/data/chinese_checkers_6_role{0}.csv",  (4, 253, 90), 6),
]

TRAIN_TEST_RATIO = 0.8 # Splits the training and test data 80/20.
learning_rate = 1e-4

# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, game_info, train_data_size, fun):
    metadata_size, state_size, num_actions = game_info
    meta = pd.read_csv(file_name, usecols = range(0,metadata_size), header = None)
    features = pd.read_csv(file_name, usecols = range(metadata_size, metadata_size + state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(metadata_size + state_size, metadata_size + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if fun(item)]

    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data =  data[cut:]
    return train_data, test_data
  
def filterData(record):
    meta, _, _ = record
    return meta[3] != 0

def read_from_csv(batch_size, train_data):
    features = []
    labels = []
    for i in range(0, batch_size):
        feature, label = random.choice(train_data)
        features.append(feature)
        labels.append(label)
    return features, labels

# -------------------------------------------------------------------
#                           Create Model                             
# -------------------------------------------------------------------


input_nets = []
input_states_dict = {}
for _, (_, state_size, _), _ in input_formats:
    input_state = tf.placeholder(tf.float32, shape=[None, state_size], name='input_state')
    
    with tf.name_scope("input_network"):
        input_net = Dense(state_size, activation='relu')(input_state)
        input_nets.append(Dense(50, activation='relu')(input_net))

# Sum all the input layers together
input_net = sum(input_nets)

# Middle Layer
with tf.name_scope("middle_network"):
    middle_net = Dense(100, activation='relu')(input_net)
   # middle_net = Dropout(0.5)(middle_net)
    middle_net = Dense(50, activation='relu')(middle_net)
    #middle_net = Dropout(0.5)(middle_net)
    middle_net = Dense(100, activation='relu')(middle_net)

role_networks = []

for file_name,game_info, num_roles in input_formats:
    _, _, num_actions = game_info
    true_action = tf.placeholder(tf.float32, shape=[None, num_actions], name='true_action')
    true_action_cls = tf.argmax(true_action, dimension=1)
    for role in range(num_roles):
        with tf.name_scope("output_network"):
            output_net = Dense(50, activation='relu')(middle_net)
            pred_action = Dense(num_actions, activation='softmax')(output_net)
            loss = tf.losses.mean_squared_error(pred_action, true_action)
        
        with tf.name_scope("accuracy"):
            pred_action_cls = tf.argmax(pred_action, dimension=1)
            correct_prediction = tf.equal(pred_action_cls, true_action_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        train_data, test_data = read_data(file_name.format(role), game_info, TRAIN_TEST_RATIO, filterData)
        print('{0} | train {1} test {2}'.format(file_name.format(role), len(train_data),len(test_data)))

        test_feed_dict = {
            input_state: [i[0] for i in test_data], 
            true_action: [i[1] for i in test_data],
            K.learning_phase(): 0
        }

        # Summaries for tensorboard
        tf.summary.scalar("loss", loss)
        tf.summary.scalar("accuracy", accuracy)

        network = ((optimizer, accuracy, loss), ( ,true_action)train_data, test_feed_dict, role)
        role_networks.append(network)


# -------------------------------------------------------------------
#                           Train Model
# -------------------------------------------------------------------

saver = tf.train.Saver()

session = tf.Session()
K.set_session(session)
session.run(tf.global_variables_initializer())
merged_summary = tf.summary.merge_all()

train_batch_size = 200

def optimize(num_iterations, roles, summary_writer):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    #best = 0
    since_last_best =0 
    for i in range(0, num_iterations):
        for (optimizer, accuracy, loss), (input_state, true_action), train_data, feed_dict_test, roleIndex in roles:
            state_batch, label_true_batch = read_from_csv(train_batch_size, train_data)
            feed_dict_train = {
                input_state: state_batch,
                true_action: label_true_batch,
                K.learning_phase(): 1
            }
            session.run(optimizer, feed_dict=feed_dict_train)

            if i % 10 == 0:
                if summary_writer != None:
                    s = session.run(merged_summary, feed_dict=feed_dict_test)
                    summary_writer.add_summary(s,i)

                train_acc = session.run(accuracy, feed_dict=feed_dict_train)
                train_loss = session.run(loss, feed_dict=feed_dict_train)
                test_acc = session.run(accuracy, feed_dict=feed_dict_test)
                test_loss = session.run(loss, feed_dict=feed_dict_test)
                #if (test_acc > best):
                #    saver.save(session, 'C:/tmp/saves/cc6-{0}'.format(i))
                #    saver.export_meta_graph('C:/tmp/saves/cc6-{0}'.format(i))    

                msg = "{0:>6}: Train Loss {1:>6.4%}  Train Correct {2:>6.1%} | Test Loss {3:>6.4%} Test Correct {4:>6.1%} Role{5}"
                print(msg.format(i + 1, train_loss, train_acc, test_loss, test_acc,roleIndex))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

summary_writer = tf.summary.FileWriter('/multygame//role1')
summary_writer.add_graph(session.graph)

#for i in range(10):
optimize(20000, role_networks, None)