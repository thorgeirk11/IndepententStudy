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

state_size = 135
num_actions = 8


# -------------------------------------------------------------------
#                           Read Data
# -------------------------------------------------------------------

def read_data(file_name, train_data_size, fun):
    # Match.connect4.1240881643605, 1, 1, 100
    meta = pd.read_csv(file_name, usecols = range(0,4), header = None)
    features = pd.read_csv(file_name, usecols = range(4,4+state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(4+state_size, 4 + state_size + num_actions), header = None)
    
    data = list(zip(meta.values, features.values, labels.values))
    data = [(item[1],item[2]) for item in data if fun(item)]

    cut = int(len(data) * train_data_size)
    train_data = data[:cut]
    test_data =  data[cut:]
    return train_data, test_data

def process_probatility(data):
    data_dict = defaultdict(list)
    for v, k in data: data_dict[tuple(v)].append(k)
    weighted_data = {}
    for key, labels in data_dict.items():
        weighed_labels = [0, 0, 0, 0, 0, 0, 0, 0]
        for label in labels:
            i = 0
            for v in label:
                if (v > 0):
                    weighed_labels[i] += 1
                    break
                i += 1
        count = len(labels)
        for i in range(0,len(weighed_labels)):
            weighed_labels[i] /= count
        weighted_data[tuple(key)] = weighed_labels
    return weighted_data

def export_processed_data(data):
    # save to file:
    with open('Connect4_train_processed.json', 'w') as f:
        for k, v in data:
            line = '{}, {}'.format(k, v) 
            print(line, file=f)     

def filterData(record):
    meta, _, _ = record
    return meta[2] == 1 and meta[3] == 100 

def guess_dumbly(data):
    correct = [0,0,0,0,0,0,0,0]
    for v, k in data:
        for i in range(0, len(k)):
            if (k[i] == 1):
                correct[i] += 1
    print(correct)
    print(len(data))

lastIndex = 0
def read_from_csv(batch_size, rand):
    global lastIndex
    features = []
    labels = []
    if rand:
        for i in range(0, batch_size):
            feature, label = random.choice(train_data)
            features.append(feature)
            labels.append(label)
        return features, labels

    trainLen = len(train_data)
    if lastIndex + batch_size < trainLen:
        end = lastIndex + batch_size
        batch = train_data[lastIndex:end]
        lastIndex = end
    else:
        rest = lastIndex + batch_size - trainLen
        batch = train_data[lastIndex:] + train_data[:rest]
        lastIndex = rest         
        
    for feature, label in batch:
        features.append(feature)
        labels.append(label)

    return features, labels


# -------------------------------------------------------------------
#                           Create Model
# -------------------------------------------------------------------



def fully_connected_network(layers, randSample):
    x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
    y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
    y_true_cls = tf.argmax(y_true, dimension=1)
    x_pretty = pt.wrap(x)
    with pt.defaults_scope(activation_fn=tf.nn.relu):
        network = x_pretty
        i = 1
        for size in layers:
            print(size)
            result = network.fully_connected(size=size, name='layer_fc{0}'.format(i))
            print(result)
            i += 1
        y_pred, loss = network.softmax_classifier(num_classes=num_actions, labels=y_true)
        if randSample:
            loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))

        y_pred_cls = tf.argmax(y_pred, dimension=1)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        optimizer = tf.train.AdamOptimizer().minimize(loss)
        name = ', '.join(str(x) for x in layers) + (' Default_Loss' if randSample else '') 
        return (name, optimizer, accuracy, ([],[]), (x, y_true), randSample)

networks = [ 
    fully_connected_network([], True)
]


    
# Splits the training and test data 80/20.
train_data, test_data = read_data('data/connect4_big.csv', 0.8, filterData)
#probatilities = process_probatility(train_data + test_data)
#guess_dumbly(train_data+test_data)


session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64

def optimize(num_iterations):
    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(0, num_iterations):
        x_batch, y_true_batch = read_from_csv(train_batch_size,True)

        for _, optimizer, _, _, (x,y_true), rndSample in networks:
            feed_dict_train = {x: x_batch,
                            y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)
        
        if i % 10 == 0:
            for name, _, accuracy, (plotx,ploty), (x, y_true), _ in networks:
                feed_dict_test = {x: [i[0] for i in test_data] ,
                                    y_true: [i[1] for i in test_data] }
                acc = session.run(accuracy, feed_dict=feed_dict_test)

                plotx.append(i)
                ploty.append(acc)

                #msg = "Optimization Iteration: {0:>6} Test Correct {1:>6.1%} Name {2}"
                #print(msg.format(i + 1, acc, name))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def argmax_label(label):
    maxi = 0
    for i in range(0, len(label)):
        if (label[i] > label[maxi]):
            maxi = i
    return maxi

optimize(5000)
for (name, (x,y)) in [(i[0], i[3]) for i in networks]:
    plt.plot(x,y,label=name)
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Optimization Iteration')
plt.title("Plot")
plt.legend()
plt.savefig('Plot.png')
plt.show()
