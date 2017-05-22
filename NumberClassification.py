
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
dir = os.path.dirname(os.path.realpath(__file__))

class NumberClassification:
    def __init__(self):
        print('')

        var_list = self.load_weights()
        self.weights = {
            'wc1': var_list[0],
            'wc2': var_list[1],
            'wd1': var_list[2],
            'out': var_list[3]
        }

        self.biases = {
            'bc1': var_list[4],
            'bc2': var_list[5],
            'bd1': var_list[6],
            'out': var_list[7]
        }

        # We use a neural net classifier
        ''' --- Our model ---'''

        #self.y = tf.nn.softmax(tf.matmul(self.x, self.W) + self.b)
        #self.vars = self.train()
        #self.trained_W = self.vars[0].astype(np.float32)
        #self.trained_b = self.vars[1].astype(np.float32)
        #print(self.trained_W.shape)
        #print(self.trained_b.shape)


    def classify_images(self, images):
        with tf.Session() as sess:
            print(sess.run(conv_net(images)))

        #print(self.x)
        #print(self.W)
        #print(self.b)
        #images = tf.cast(images.reshape(images.shape[0], -1), tf.float32)
        #print(images.shape)
        #labels = None
        #y_model = tf.nn.softmax(tf.matmul(images, self.trained_W) + self.trained_b)
        #with tf.Session() as sess:
            #labels = sess.run(y_model)
        #print(labels)
        #print(labels.shape)


    def conv_net(x):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])

        # Convolution Layer
        conv1 = conv2d(x, self.weights['wc1'], self.biases['bc1'])
        # Max Pooling (down-sampling)
        conv1 = maxpool2d(self.conv1, k=2)

        # Convolution Layer
        conv2 = conv2d(self.conv1, self.weights['wc2'], self.biases['bc2'])
        # Max Pooling (down-sampling)
        conv2 = maxpool2d(conv2, k=2)

        # Fully connected layer
        # Reshape conv2 output to fit fully connected layer input
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].get_shape().as_list()[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)
        # Apply Dropout
        #fc1 = tf.nn.dropout(fc1, self.dropout)

        # Output, class prediction
        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    def load_weights(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(dir + '/vars.ckpt.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./'))
            sess.run(tf.global_variables_initializer())
            graph = tf.get_default_graph()
            wc1 = graph.get_tensor_by_name('wc1:0')
            wc2 = graph.get_tensor_by_name('wc2:0')
            wd1 = graph.get_tensor_by_name('wd1:0')
            w_out = graph.get_tensor_by_name('w_out:0')
            bc1 = graph.get_tensor_by_name('bc1:0')
            bc2 = graph.get_tensor_by_name('bc2:0')
            bd1 = graph.get_tensor_by_name('bd1:0')
            b_out = graph.get_tensor_by_name('b_out:0')
            return [wc1, wc2, wd1, w_out, bc1, bc2, bd1, b_out]

    # def train(self):
    #     var_list = []
    #     y_ = tf.placeholder(tf.float32, [None, 10]) # Correct labels
    #     cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(self.y), reduction_indices=[1]))
    #     train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
    #     with tf.Session() as sess:
    #         sess.run(tf.global_variables_initializer())
    #         for _ in range(1000):
    #             batch_xs, batch_ys = self.data.train.next_batch(100)
    #             sess.run(train_step, feed_dict={self.x: batch_xs, y_: batch_ys})
    #         correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
    #         accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #         print(sess.run(accuracy, feed_dict={self.x: self.data.test.images, y_: self.data.test.labels}))
    #         for v in tf.trainable_variables():
    #             var_list.append(v.eval())
    #     return var_list
    #
    #
    # def read_in_dataset(self):
    #     data = input_data.read_data_sets("MNIST_data/", one_hot=True)
    #     print('Data loaded...')
    #     return data
