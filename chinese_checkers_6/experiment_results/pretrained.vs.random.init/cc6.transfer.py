# %matplotlib inline

import tensorflow as tf
import numpy as np
#from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math
import numpy

# We also need PrettyTensor.
import prettytensor as pt
import pandas as pd
import random
from NetworkConstruction import NetworkConstruction as nc

metadata_size = 4
state_size = 253
num_actions = 90

train_batch_size = 200

DATA_PATH = 'data/chinese_checkers_6_role{0}_noNoop.csv'
TEST_RATIO = 0.8  # Splits the training and test data 80/20
LEARNING_RATE = 1e-3


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
    return meta[3] > 0

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

x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)


input_net = nc.input_network([state_size], x)
middle_net = nc.middle_network([50,100,50], input_net)

output_layers = []

role_networks = []

for i in range(0,6):
    # Input Data
    train_data, test_data = read_data(DATA_PATH.format(i), TEST_RATIO, filterData)
    test_feed_dict = {x: [i[0] for i in test_data], y_true: [i[1] for i in test_data] }
    print('Role {0} data: train {1} test {2}'.format(i, len(train_data),len(test_data)))

    # Output Layer Construction
    output_net = nc.output_network(output_layers, middle_net, 'role{0}'.format(i),)

    # Create the sofmax 
    last_layer = nc.softmax_adamOptimizer(output_net, y_true, LEARNING_RATE,  0 == i, 'role{0}'.format(i))
    network_info = (last_layer, train_data, test_feed_dict, i)
    
    role_networks.append(network_info)


# -------------------------------------------------------------------
#                           Train Model
# -------------------------------------------------------------------

saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())



def optimize(num_iterations, roles, writer):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    #best = 0
    since_last_best =0 
    for i in range(0, num_iterations):
        roleIndex = 0
        for (optimizer, accuracy, loss), train_data, feed_dict_test, roleIndex in roles:
            x_batch, y_true_batch = read_from_csv(train_batch_size, train_data)
            feed_dict_train = {x: x_batch,
                                y_true: y_true_batch}
            session.run(optimizer, feed_dict=feed_dict_train)

            if i % 10 == 0:
                if writer != None:
                    s = session.run(merged_summary, feed_dict=feed_dict_test)
                    writer.add_summary(s,i)

                train_acc = session.run(accuracy, feed_dict=feed_dict_train)
                train_loss = session.run(loss, feed_dict=feed_dict_train)
                test_acc = session.run(accuracy, feed_dict=feed_dict_test)
                test_loss = session.run(loss, feed_dict=feed_dict_test)
                #if (test_acc > best):
                #    saver.save(session, 'C:/tmp/saves/cc6-{0}'.format(i))
                #    saver.export_meta_graph('C:/tmp/saves/cc6-{0}'.format(i))    

                msg = "{0:>6}: Train Loss {1:>6.1%}  Train Correct {2:>6.1%} | Test Loss {3:>6.1%} Test Correct {4:>6.1%} Role{5}"
                print(msg.format(i + 1, train_loss, train_acc, test_loss, test_acc,roleIndex))

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))



merged_summary = tf.summary.merge_all()
#pretrained_writer = tf.summary.FileWriter('/tansfer/logs/8/pretrained_init')
#pretrained_writer.add_graph(session.graph)
random_writer = tf.summary.FileWriter('/tansfer/logs/8/random_init')
random_writer.add_graph(session.graph)


#optimize(2000, role_networks[1:], None)
optimize(750, role_networks[:1], random_writer)