
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import cv2
import numpy as np
import tensorflow as tf

dir = os.path.dirname(os.path.realpath(__file__))

class NumberClassification:
    def __init__(self):
        print('')
        self.weights = None
        self.biases = None
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


    def classify_images(self, images):
        print(images.shape)
        images = tf.cast(images, tf.float32)
        with tf.Session() as sess:
            pred = sess.run(self.conv_net(images))
            print(len(pred))
            return pred

        # #batch_size = 10
        # #images = tf.cast(images, tf.float32)
        #
        # batch_x, batch_y = mnist.train.next_batch(batch_size)
        # #print('pred ->', batch_y, '\n--\n')
        # cv2.imshow('img', batch_x[0])
        # with tf.Session() as sess:
        #     return sess.run(self.conv_net(batch_x)), batch_y

    def maxpool2d(self, x, k=2):
        return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME')

    def conv2d(self, x, W, b, strides=1):
        x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
        x = tf.nn.bias_add(x, b)
        return tf.nn.relu(x)

    def conv_net(self, x):
        # Reshape input picture
        x = tf.reshape(x, shape=[-1, 28, 28, 1])        

        conv1 = self.conv2d(x, self.weights['wc1'], self.biases['bc1'])
        conv1 = self.maxpool2d(conv1, k=2)

        conv2 = self.conv2d(conv1, self.weights['wc2'], self.biases['bc2'])
        conv2 = self.maxpool2d(conv2, k=2)

        # Fully connected layer
        fc1 = tf.reshape(conv2, [-1, self.weights['wd1'].shape[0]])
        fc1 = tf.add(tf.matmul(fc1, self.weights['wd1']), self.biases['bd1'])
        fc1 = tf.nn.relu(fc1)

        out = tf.add(tf.matmul(fc1, self.weights['out']), self.biases['out'])
        return out

    def load_weights(self):
        with tf.Session() as sess:
            saver = tf.train.import_meta_graph(dir + '/vars.ckpt.meta')
            saver.restore(sess,tf.train.latest_checkpoint('./'))
            graph = tf.get_default_graph()
            wc1 = graph.get_tensor_by_name('wc1:0').eval()
            wc2 = graph.get_tensor_by_name('wc2:0').eval()
            wd1 = graph.get_tensor_by_name('wd1:0').eval()
            w_out = graph.get_tensor_by_name('w_out:0').eval()
            bc1 = graph.get_tensor_by_name('bc1:0').eval()
            bc2 = graph.get_tensor_by_name('bc2:0').eval()
            bd1 = graph.get_tensor_by_name('bd1:0').eval()
            b_out = graph.get_tensor_by_name('b_out:0').eval()
            return [wc1, wc2, wd1, w_out, bc1, bc2, bd1, b_out]
