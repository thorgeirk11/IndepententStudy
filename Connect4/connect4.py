# %matplotlib inline

import csv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
from collections import defaultdict 
import numpy

# We also need PrettyTensor.
import prettytensor as pt
import pandas as pd
import random

metadata_size = 4
state_size = 135
num_actions = 8

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
    return meta[2] == 0 and meta[3] == 100

def read_from_csv(batch_size):
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

x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def fully_connected_network(layers, name):
    x_pretty = pt.wrap(x)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        network = x_pretty
        i = 1
        for size in layers:
            network = network.fully_connected(size=size, name='{0}_layer_fc{1}_{2}'.format(name, i, size))
            i += 1
        with tf.name_scope("{0}_softmax".format(name)):
            y_pred, _ = network.softmax(labels=y_true)
            loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
            tf.summary.scalar("{0}_softmax".format(name), loss)

        with tf.name_scope("{0}_accuracy".format(name)):
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("{0}_accuracy".format(name), accuracy)

        with tf.name_scope("{0}_train".format(name)):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

        name = ', '.join(str(x) for x in layers[:len(layers)-1]) 
        return (name, optimizer, accuracy)

network = fully_connected_network([128,64,128,64,128,8], "A")
    
# Splits the training and test data 80/20.
train_data, test_data = read_data('data/connect4_big.csv', 0.8, filterData)

# -------------------------------------------------------------------
#                           Train Model
# -------------------------------------------------------------------

session = tf.Session()
session.run(tf.global_variables_initializer())

merged_summary = tf.summary.merge_all()
file_writer = tf.summary.FileWriter('/logs/5')
file_writer.add_graph(session.graph)

train_batch_size = 64

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    best = 0
    since_last_best =0 
    feed_dict_test = {x: [i[0] for i in test_data] ,
                      y_true: [i[1] for i in test_data] }
    i = 0
    while True:
        i+=1
        x_batch, y_true_batch = read_from_csv(train_batch_size)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        _, optimizer, _ = network
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 5 == 0:
            #for name, _, accuracy in networks:
            s = session.run(merged_summary, feed_dict=feed_dict_test)
            file_writer.add_summary(s,i)
                #acc = session.run(accuracy, feed_dict=feed_dict_test)
                #if (acc > best):
                #    best = acc
                #    since_last_best = 0

                #msg = "Optimization Iteration: {0:>6} Test Correct {1:>6.1%} Best {2}"
                #print(msg.format(i + 1, acc, best))
            
        since_last_best += 1
        if since_last_best > num_iterations:
            break

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(8000)