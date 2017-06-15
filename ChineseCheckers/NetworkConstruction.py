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
        with tf.name_scope("middle_network"):
            i = 1
            for size in layers:
                network = input_network.fully_connected(size=size, name='middle_fc{0}'.format(i), activation_fn=tf.nn.relu)
                i += 1
        return network

    @staticmethod
    def output_network(layers, middle_network):
        with tf.name_scope("output_network"):
            i = 1
            for size in layers:
                network = middle_network.fully_connected(size=size, name='output_fc{0}'.format(i), activation_fn=tf.nn.relu)
                i += 1
        return network
        
    @staticmethod
    def classifier(network, learning_rate):
        with tf.name_scope("loss"):
            y_pred, loss = network.softmax(labels=y_true)
            tf.summary.scalar("loss", loss)

        with tf.name_scope("accuracy"):
            y_pred_cls = tf.argmax(y_pred, dimension=1)
            correct_prediction = tf.equal(y_pred_cls, y_true_cls)
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            tf.summary.scalar("accuracy", accuracy)

        with tf.name_scope("train"):
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

        return optimizer, accuracy, loss