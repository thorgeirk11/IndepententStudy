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

# Splits the training and test data 80/20.
train_data_role0, test_data_role0 = read_data('data/chinese_checkers_6_role0_noNoop.csv', 0.8, filterData)
train_data_role1, test_data_role1 = read_data('data/chinese_checkers_6_role1_noNoop.csv', 0.8, filterData)
print('Role 0 data: train {0} test {1}'.format(len(train_data_role0),len(test_data_role0)))
print('Role 1 data: train {0} test {1}'.format(len(train_data_role1),len(test_data_role1)))

# -------------------------------------------------------------------
#                           Create Model
# -------------------------------------------------------------------


x = tf.placeholder(tf.float32, shape=[None, state_size], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_actions], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

input_net = nc.input_network([state_size,500], x)
middle_net = nc.middle_network([5000], input_net)
output_net = nc.output_network([500,num_actions], middle_net)

optimizer, accuracy, loss = classifier(output_net, 1e-3)


# -------------------------------------------------------------------
#                           Train Model
# -------------------------------------------------------------------

saver = tf.train.Saver()

session = tf.Session()
session.run(tf.global_variables_initializer())

#merged_summary = tf.summary.merge_all()
#file_writer = tf.summary.FileWriter('/logs/5')
#file_writer.add_graph(session.graph)

train_batch_size = 200

def optimize(num_iterations, train_data, test_data):
    # Start-time used for printing time-usage below.
    start_time = time.time()
    best = 0
    since_last_best =0 
    feed_dict_test = {x: [i[0] for i in test_data] ,
                      y_true: [i[1] for i in test_data] }
    i = 0
    while True:
        x_batch, y_true_batch = read_from_csv(train_batch_size, train_data)

        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        session.run(optimizer, feed_dict=feed_dict_train)

        if i % 10 == 0:
            #s = session.run(merged_summary, feed_dict=feed_dict_test)
            #file_writer.add_summary(s,i)
            train_acc = session.run(accuracy, feed_dict=feed_dict_train)
            train_loss = session.run(loss, feed_dict=feed_dict_train)
            test_acc = session.run(accuracy, feed_dict=feed_dict_test)
            test_loss = session.run(loss, feed_dict=feed_dict_test)
            if (test_acc > best):
                saver.save(session, 'C:/tmp/saves/cc6-{0}'.format(i))
                saver.export_meta_graph('C:/tmp/saves/cc6-{0}'.format(i))    
                best = test_acc
                since_last_best = 0

            msg = "{0:>6}: Train Loss {1:>6.1%}  Train Correct {2:>6.1%} | Test Loss {3:>6.1%} Test Correct {4:>6.1%} Best {5:>6.3%}"
            print(msg.format(i + 1, train_loss, train_acc, test_loss, test_acc, best))
            
        since_last_best += 1
        if since_last_best > num_iterations:
            break
            
        i+=1

    end_time = time.time()
    time_dif = end_time - start_time
    print("Time usage: " + str(timedelta(seconds=int(round(time_dif)))))


optimize(2500, train_data_role1, test_data_role1)