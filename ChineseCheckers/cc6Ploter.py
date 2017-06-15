import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
import math
from collections import defaultdict 
import numpy

# We also need PrettyTensor.
import prettytensor as pt
import pandas as pd
import random

LOG_PATH = '/logs/cc6/cmp20/'

num_iterations = 10000

metadata_size = 4
state_size = 253
num_actions = 90

# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, train_data_size, fun):
    meta = pd.read_csv(file_name, usecols = range(0,metadata_size), header = None)
    features = pd.read_csv(file_name, usecols = range(metadata_size,metadata_size+state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(metadata_size + state_size, metadata_size + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if fun(item)]

    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data =  data[cut:]
    return train_data, test_data
  
def filterData(record):
    meta, _, _ = record
    return meta[2] == 0 and meta[3] > 0

def read_from_csv(batch_size):
    features = []
    labels = []
    for i in range(0, batch_size):
        feature, label = random.choice(train_data)
        features.append(feature)
        labels.append(label)
    return features, labels
    
# Splits the training and test data 80/20.
train_data, test_data = read_data('data/chinese_checkers_6_role0_noNoop.csv', 0.8, filterData)

# -------------------------------------------------------------------
#                           Create Model
# -------------------------------------------------------------------


def BuildAndTrain(layers, learning_rate, activation):
    tf.reset_default_graph()
    session = tf.Session()

    x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)

    hparam = '_'.join(str(x) for x in layers[:len(layers)]) + '_LR_' + str(learning_rate) + '_' + ('none' if activation == None else "relu")
   
    # -------------------------------------------------------------------
    #                           Pretty Network
    # -------------------------------------------------------------------
    x_pretty = pt.wrap(x)
    with pt.defaults_scope(activation_fn=activation):
        network = x_pretty
        i = 1
        for size in layers:
            network = network.fully_connected(size=size, name='layer_fc{0}_{1}'.format(i, size))
            i += 1

        with tf.name_scope("loss"):     
            y_pred, loss = network.softmax_classifier(num_classes=num_actions, labels=y_true)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("accuracy"):
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    
    # -------------------------------------------------------------------
    #                           Train Model
    # -------------------------------------------------------------------

    train_batch_size = 64

    session.run(tf.global_variables_initializer())

    merged_summary = tf.summary.merge_all()
    test_writer = tf.summary.FileWriter(LOG_PATH  + hparam)
    test_writer.add_graph(session.graph)

    #train_writer = tf.summary.FileWriter(LOG_PATH + '/train/' + hparam)
    #train_writer.add_graph(session.graph)
    
    #Start-time used for printing time-usage below.
    start_time = time.time()

    feed_dict_test = {x: [i[0] for i in test_data] ,
                    y_true: [i[1] for i in test_data] }
    for i in range(0, num_iterations):
        x_batch, y_true_batch = read_from_csv(train_batch_size)

        feed_dict_train = {x: x_batch,
                            y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)
        if i % 10 == 0:
            #s, train_acc = session.run([merged_summary, accuracy], feed_dict=feed_dict_train)
            #train_writer.add_summary(s,i)
            
            s, test_acc = session.run([merged_summary, accuracy], feed_dict=feed_dict_test)
            test_writer.add_summary(s,i)

        #msg = "Optimization Iteration: {0:>6} Test Correct {1:>6.1%} Best {2}"
        #print(msg.format(i + 1, train_acc, best))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

networks = [
    BuildAndTrain([], learning_rate=1e-4),
    BuildAndTrain([10000], learning_rate=1e-5),
    BuildAndTrain([10000], learning_rate=1e-4),
    BuildAndTrain([1000,250,500,250,1000], learning_rate=1e-3),
    BuildAndTrain([500,250,500,250,100], learning_rate=1e-4),
    BuildAndTrain([500,250,500,250,100], learning_rate=1e-3),
]