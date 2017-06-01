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
num_actions = 7

x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, loss = x_pretty.\
        fully_connected(size=state_size*2, name='layer_fc1').\
        fully_connected(size=state_size, name='layer_fc2').\
        fully_connected(size=50, name='layer_fc3').\
        fully_connected(size=state_size, name='layer_fc4').\
        fully_connected(size=num_actions, name='layer_fc5').\
        softmax_classifier(num_classes=num_actions, labels=y_true)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(loss)

y_pred_cls = tf.argmax(y_pred, dimension=1)

correct_prediction = tf.equal(y_pred_cls, y_true_cls)

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def read_data(file_name):
    features = pd.read_csv(file_name, usecols = range(0,state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(state_size, state_size + num_actions), header = None)
    data = zip(features.values, labels.values)
    data_dict = defaultdict(list)
    for v, k in data: data_dict[tuple(v)].append(k)

def 
    weighted_data = []
    for key, labels in data_dict.items():
        weighed_labels = [0, 0, 0, 0, 0, 0, 0]
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
        weighted_data.append( (list(key),weighed_labels) )

    return weighted_data

train_data, test_data = read_data('Connect4_data_role1.csv.csv')


# save to file:
with open('Connect4_train_processed.json', 'w') as f:
    for k, v in train_data:
        line = '{}, {}'.format(k, v) 
        print(line, file=f)     

print(len(train_data))
print(len(test_data))
def read_from_csv(batch_size):
    features = []
    labels = []
    for i in range(0, batch_size):
        feature, label = random.choice(train_data)
        features.append(feature)
        labels.append(label)
    return features, labels


session = tf.Session()
session.run(tf.global_variables_initializer())

train_batch_size = 64
# Counter for total number of iterations performed so far.
total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch = read_from_csv(train_batch_size)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc_train = session.run(accuracy, feed_dict=feed_dict_train)
            
            feed_dict = {x: [i[0] for i in test_data] ,
                            y_true: [i[1] for i in test_data] }

            acc = session.run(accuracy, feed_dict=feed_dict)
            
            example = test_data[15]
            feed_dict_2 = {x: [example[0]],
                            y_true: [example[1]] }
            predictied = session.run(y_pred, feed_dict= feed_dict_2)
            l = session.run(loss, feed_dict= feed_dict_2)
            print(list(predictied[0]))
            print(example[1])
            print(l)


            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Train Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%}"
            
            # Print it.
            print(msg.format(i + 1, acc_train, acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(3000)
