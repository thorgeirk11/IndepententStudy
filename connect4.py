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

x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

#x_pretty = pt.wrap(x)
#with pt.defaults_scope(activation_fn=tf.nn.relu):
#    y_pred, loss = x_pretty.\
#        fully_connected(size=256, name='layer_fc1').\
#        fully_connected(size=128, name='layer_fc2').\
#        fully_connected(size=64, name='layer_fc3').\
#        fully_connected(size=128, name='layer_fc4').\
#        fully_connected(size=128, name='layer_fc1').\
#        fully_connected(size=64, name='layer_fc1').\
#        softmax_classifier(num_classes=num_actions, labels=y_true)
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer
    
def CreateNetwork(networkNodeNumbers):
    layer = new_fc_layer(input=x,
                            num_inputs=state_size,
                            num_outputs=networkNodeNumbers[0],
                            use_relu=True)

    for i in range(0, len(networkNodeNumbers) - 1):
        layer = new_fc_layer(input=layer,
                             num_inputs=networkNodeNumbers[i],
                             num_outputs=networkNodeNumbers[i+1],
                             use_relu=True)
                             
    layer = new_fc_layer(input=layer,
                            num_inputs=networkNodeNumbers[len(networkNodeNumbers)-1],
                            num_outputs=num_actions,
                            use_relu=False)

    # Predicted class-label.
    y_pred = tf.nn.softmax(layer)
    
    # Loss aka. cost-measure.
    # This is the scalar value that must be minimized.
    loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
    return y_pred, loss

y_pred, loss = CreateNetwork([128,64,128,64,128])

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)
y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def read_data(file_name, train_data_size):
    features = pd.read_csv(file_name, usecols = range(0,state_size), header = None)
    labels  = pd.read_csv(file_name, usecols = range(state_size, state_size + num_actions), header = None)
    data = list(zip(features.values, labels.values))
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

# Splits the training and test data 80/20.
train_data, test_data = read_data('data/Connect4_data_role1.csv', 0.6)
probatilities = process_probatility(train_data + test_data)


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

    feed_dict_test = {x: [i[0] for i in test_data] ,
                      y_true: [i[1] for i in test_data] }
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
            acc_train = session.run(loss, feed_dict=feed_dict_train)
            acc = session.run(loss, feed_dict=feed_dict_test)
            
            #key = (1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1)
                #[1.9460331e-05, 0.042004984, 0.058386341, 0.095144488, 0.60808468, 0.11274167, 0.061282061, 0.022336284]
                #[0.0, 0.10964912280701754, 0.14912280701754385, 0.10526315789473684, 0.5087719298245614, 0.06140350877192982, 0.04824561403508772, 0.017543859649122806]
            #key = (1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1)
                #[8.5227482e-05, 0.032720458, 0.078245468, 0.1244887, 0.5727576, 0.10703135, 0.066311382, 0.018359911]
                #[0.0, 0.14792899408284024, 0.029585798816568046, 0.13609467455621302, 0.5088757396449705, 0.1203155818540434, 0.04930966469428008, 0.007889546351084813]
            #key = (1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1,1,-1,-1)
                #[0.0002040648, 0.044207145, 0.086732984, 0.16322351, 0.44881546, 0.14300202, 0.079944506, 0.033870406]
                #[0.0, 0.10454545454545454, 0.06818181818181818, 0.18636363636363637, 0.5318181818181819, 0.05, 0.022727272727272728, 0.03636363636363636]
            #label = probatilities[key]
            #feed_dict_2 = {x: [list(key)],
            #                y_true: [label] }
            #predictied = session.run(y_pred, feed_dict= feed_dict_2)
            #error = session.run(loss, feed_dict= feed_dict_2)
            #print(list(predictied[0]))
            #print(label)
            #print(error)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Train Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%}"
            
            # Print it.
            print(msg.format(i + 1, 1-acc_train, 1-acc))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

optimize(30000)
