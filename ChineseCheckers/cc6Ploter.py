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

# -------------------------------------------------------------------
#                           Create Model
# -------------------------------------------------------------------

x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def fully_connected_network(layers):
    x_pretty = pt.wrap(x)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        network = x_pretty
        i = 1
        for size in layers:
            #with tf.name_scope('fully_connected'):
            network = network.fully_connected(size=size, name='layer_fc{0}_{1}'.format(i, size))
            i += 1
        with tf.name_scope("softmax"):
            y_pred, _ = network.softmax_classifier(num_classes=num_actions, labels=y_true)
            loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))

        with tf.name_scope("accuracy"):
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

        name = ', '.join(str(x) for x in layers) 
        return (name, optimizer, accuracy, ([],[]), (y_pred,loss))


networks = [ 
    fully_connected_network([]),
]


    
# Splits the training and test data 80/20.
train_data, test_data = read_data('data/chinese_checkers_6_role0_noNoop.csv', 0.8, filterData)

# -------------------------------------------------------------------
#                           Train Model
# -------------------------------------------------------------------

session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    feed_dict_test = {x: [i[0] for i in test_data] ,
                      y_true: [i[1] for i in test_data] }
    for i in range(0, num_iterations):
        x_batch, y_true_batch = read_from_csv(train_batch_size)

        feed_dict_train = {x: x_batch,
                            y_true: y_true_batch}
        for _, optimizer, _, _, rndSample in networks:
            session.run(optimizer, feed_dict=feed_dict_train_rnd)

        if i % 10 == 0:
            for name, _, accuracy, (plotx,ploty), _ in networks:
                acc = session.run(accuracy, feed_dict=feed_dict_test)

                plotx.append(i)
                ploty.append(acc)

                msg = "Optimization Iteration: {0:>6} Test Correct {1:>6.1%} Name {2}"
                print(msg.format(i + 1, acc, name))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(2000)

for (name, (x,y)) in [(i[0], i[3]) for i in networks]:
    plt.plot(x,y,label=name)
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Optimization Iteration')
plt.title("Plot")
plt.legend()
plt.savefig('Plot.png')
plt.show()
