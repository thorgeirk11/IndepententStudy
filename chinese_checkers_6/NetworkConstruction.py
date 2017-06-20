import tensorflow as tf
import prettytensor as pt

class NetworkConstruction(object):
    @staticmethod
    def input_network(layers, input_tensor):
        network = pt.wrap(input_tensor)
        with tf.name_scope("input_network"):
            i = 1
            for size in layers:
                network = network.fully_connected(size=size, name='input_fc{0}'.format(i), activation_fn=tf.nn.relu)
                i += 1
        return network
    @staticmethod
    def middle_network(layers, input_network):
        network = input_network
        with tf.name_scope("middle_network"):
            i = 1
            for size in layers:
                network = network.fully_connected(size=size, name='middle_fc{0}'.format(i), activation_fn=tf.nn.relu)
                i += 1
        return network

    @staticmethod
    def output_network(layers, middle_network, name):
        network = middle_network
        with tf.name_scope("output_network_{0}".format(name)):
            i = 1
            for size in layers:
                network = network.fully_connected(size=size, name='output_fc{0}_{1}'.format(i, name), activation_fn=tf.nn.relu)
                i += 1
        return network
        
    @staticmethod
    def softmax_adamOptimizer(network, true_labels, learning_rate, name):
        with tf.name_scope("loss"):
            num_actions = int(true_labels.get_shape()[1])
            pred_labels, loss = network.softmax_classifier(num_classes=num_actions, labels=true_labels, name="softmax_{0}".format(name))
            tf.summary.scalar("loss", loss)

        with tf.name_scope("accuracy"):
            pred_cls = tf.argmax(pred_labels, dimension=1)
            true_cls = tf.argmax(true_labels, dimension=1)
            correct_prediction = tf.equal(pred_cls, true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        return optimizer, accuracy, loss