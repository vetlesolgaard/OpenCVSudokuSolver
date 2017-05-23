
from __future__ import print_function

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import cv2
import numpy as np
import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

dir = os.path.dirname(os.path.realpath(__file__))

# Parameters
learning_rate = 0.001
training_iters = 50000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.75 # Dropout, probability to keep units

# tf Graph input
x = tf.placeholder(tf.float32, [None, n_input])
y = tf.placeholder(tf.float32, [None, n_classes])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

def load_weights():
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

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

def conv_net2(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 28, 28, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].shape[0]])
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    out = tf.add(tf.matmul(fc1, weights['out']), biases['out'])
    return out

# Store layers weight & bias
weights = {
    # 5x5 conv, 1 input, 32 outputs
    'wc1': tf.Variable(tf.random_normal([5, 5, 1, 32]), name='wc1'),
    # 5x5 conv, 32 inputs, 64 outputs
    'wc2': tf.Variable(tf.random_normal([5, 5, 32, 64]), name='wc2'),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(tf.random_normal([7*7*64, 1024]), name='wd1'),
    # 1024 inputs, 10 outputs (class prediction)
    'out': tf.Variable(tf.random_normal([1024, n_classes]), name='w_out')
}

biases = {
    'bc1': tf.Variable(tf.random_normal([32]), name='bc1'),
    'bc2': tf.Variable(tf.random_normal([64]), name='bc2'),
    'bd1': tf.Variable(tf.random_normal([1024]), name='bd1'),
    'out': tf.Variable(tf.random_normal([n_classes]), name='b_out')
}

# Construct model
pred = conv_net(x, weights, biases, keep_prob)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                       keep_prob: dropout})
        if step % display_step == 0:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                              y: batch_y,
                                                              keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 256 mnist test images
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={x: mnist.test.images[:256],
                                      y: mnist.test.labels[:256],
                                      keep_prob: 1.}))

    saver = tf.train.Saver()
    path = saver.save(sess, dir + '/vars.ckpt')
    for v in tf.trainable_variables():
        print(v.name)

    batch_x = mnist.test.images[:10]
    batch_y = mnist.test.labels[:10]
    # print(batch_x[0])
    # img = np.reshape(batch_x[0], (28,28))
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    out = sess.run(conv_net(batch_x, weights, biases, 1.0))
    for i in range(0, len(batch_y)):
        print(np.argmax(batch_y[i]), np.argmax(out[i]))
    print(sess.run(biases['bc1']))
# Let's load a previously saved meta graph in the default graph
# This function returns a Saver

''' Save Variables'''
# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph(dir + '/vars.ckpt-128.meta')
#     saver.restore(sess,tf.train.latest_checkpoint('./'))
#     graph = tf.get_default_graph()
#     bc1 = graph.get_tensor_by_name('bc1:0').eval()
#     print(bc1)

#
print('\nLOAD TEST\n')
# var_list = load_weights()
# #
# new_weights = {
#     'wc1': var_list[0],
#     'wc2': var_list[1],
#     'wd1': var_list[2],
#     'out': var_list[3]
# }
# new_biases = {
#     'bc1': var_list[4],
#     'bc2': var_list[5],
#     'bd1': var_list[6],
#     'out': var_list[7]
# }
#
# batch_size = 10
# batch_x = mnist.test.images[:10]
# batch_y = mnist.test.labels[:10]
# with tf.Session() as sess:
#     out = sess.run(conv_net2(batch_x, new_weights, new_biases, 1.0))
#     print(out.shape)
#     print(batch_y.shape)
#     for i in range(0, len(batch_y)):
#         print(np.argmax(batch_y[i]), np.argmax(out[i]))
#         cv2.imshow('img', batch_x[i])
