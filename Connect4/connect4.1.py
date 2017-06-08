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

x_pretty = pt.wrap(x)
with pt.defaults_scope(activation_fn=tf.nn.relu):
    y_pred, _ = x_pretty.\
        fully_connected(size=128, name='layer_fc2').\
        fully_connected(size=64, name='layer_fc3').\
        fully_connected(size=128, name='layer_fc4').\
        fully_connected(size=64, name='layer_fc3').\
        fully_connected(size=128, name='layer_fc4').\
        softmax_classifier(num_classes=num_actions, labels=y_true)
loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))


y_pred_cls = tf.argmax(y_pred, dimension=1)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(loss)

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
    meta, key, label = record
    return meta[2] == 1 and meta[3] == 100 
    
# Splits the training and test data 80/20.
train_data, test_data = read_data('data/Connect4_data_Step&BothRoles2.csv', 0.8, filterData)
probatilities = process_probatility(train_data + test_data)



def guess_dumbly(data):
    correct = [0,0,0,0,0,0,0,0]
    for v, k in data:
        for i in range(0, len(k)):
            if (k[i] == 1):
                correct[i] += 1
    print(correct)
    print(len(data))
    
guess_dumbly(train_data+test_data)


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
    
    plotx = []
    ploty = []
 
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
        #if i % 1000 == 0:
        # Calculate the accuracy on the training-set.
        #acc_train = session.run(loss, feed_dict=feed_dict_train)
        #acc = session.run(loss, feed_dict=feed_dict_test)
        acc2 = session.run(accuracy, feed_dict=feed_dict_test)

        #np.set_printoptions(precision=2)
        #label = probatilities[key]
        #feed_dict_2 = {x: [list(key)],
        #                y_true: [label] }
        #key, label = key, value = probatilities.popitem()
        #while (argmax_label(label) == 0):
        #    key, label = key, value = probatilities.popitem()

        #feed_dict_2 = {x: [list(key)],
        #                y_true: [label] }

        #predictied = session.run(y_pred, feed_dict= feed_dict_2)
        #tf.argmax(predictied, dimension=1)
        #error = session.run(loss, feed_dict= feed_dict_2)
        #error = session.run(loss, feed_dict= feed_dict_2)

        #print("Predicted {0}".format(predictied[0]))
        #print("Was {0}".format(label))

        #print(list(predictied[0]))
        #print("Predicted {0} was {1}".format(session.run(tf.argmax(predictied, dimension=1))[0],argmax_label(label)))
        #print(error)

        # Message for printing.
        msg = "Optimization Iteration: {0:>6}, Train Accuracy: {1:>6.1%}, Test Accuracy: {2:>6.1%} Test Correct {3:>6.1%}"

        # Print it.
        print(msg.format(i + 1, 1-acc_train, 1-acc, acc2))

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))

def argmax_label(label):
    maxi = 0
    for i in range(0, len(label)):
        if (label[i] > label[maxi]):
            maxi = i
    return maxi

optimize(30000)


def print_confusion_matrix():
    # Get the true classifications for the test-set.
    cls_true = data.test.cls
    
    # Get the predicted classifications for the test-set.
    cls_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)

    # Get the confusion matrix using sklearn.
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)

    # Print the confusion matrix as text.
    print(cm)

    # Plot the confusion matrix as an image.
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    # Make various adjustments to the plot.
    plt.tight_layout()
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')




























    
# def new_biases(length):
#     return tf.Variable(tf.constant(0.05, shape=[length]))
# 
# def new_weights(shape):
#     return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
# 
# def new_fc_layer(input,          # The previous layer.
#                  num_inputs,     # Num. inputs from prev. layer.
#                  num_outputs,    # Num. outputs.
#                  use_relu=True): # Use Rectified Linear Unit (ReLU)?
# 
#     # Create new weights and biases.
#     weights = new_weights(shape=[num_inputs, num_outputs])
#     biases = new_biases(length=num_outputs)
# 
#     # Calculate the layer as the matrix multiplication of
#     # the input and weights, and then add the bias-values.
#     layer = tf.matmul(input, weights) + biases
# 
#     # Use ReLU?
#     if use_relu:
#         layer = tf.nn.relu(layer)
# 
#     return layer
#     
# def CreateNetwork(networkNodeNumbers):
#     layer = new_fc_layer(input=x,
#                             num_inputs=state_size,
#                             num_outputs=networkNodeNumbers[0],
#                             use_relu=True)
# 
#     for i in range(0, len(networkNodeNumbers) - 1):
#         layer = new_fc_layer(input=layer,
#                              num_inputs=networkNodeNumbers[i],
#                              num_outputs=networkNodeNumbers[i+1],
#                              use_relu=True)
#                              
#     layer = new_fc_layer(input=layer,
#                             num_inputs=networkNodeNumbers[len(networkNodeNumbers)-1],
#                             num_outputs=num_actions,
#                             use_relu=False)
# 
#     # Predicted class-label.
#     y_pred = tf.nn.softmax(layer)
#     
#     # Loss aka. cost-measure.
#     # This is the scalar value that must be minimized.
#     loss = tf.reduce_mean(tf.squared_difference(y_pred, y_true))
#     return y_pred, loss
# 
# y_pred, loss = CreateNetwork([128,64,128])