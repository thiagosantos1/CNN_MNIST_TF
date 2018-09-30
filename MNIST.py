import tensorflow as tf
import pandas as pad
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


# create a weight variable - Filter
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1) # to init with a random normal distribution with stndard deviation of 1
  return tf.Variable(initial) # create a variable

# create a bias variable
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# execute and return a convolutional over a data x, with a filter/weights W
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# create, execute and return the result of a convolutional layer, already with an activation 
def conv_layer(input, shape):
  W = weight_variable(shape)
  b = bias_variable([shape[3]])
  return tf.nn.relu(conv2d(input, W) + b)

# with X as a result of a convolutional layer, we will max pool
# with a filter of 2x2
# this basically gets the most important features, and reduces the size of the inputs for the final densed layer
def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
  strides=[1, 2, 2, 1], padding='SAME')


# After all convolutional layers has been applied, we get all the final results, and make a full connected layer
def full_layer(input, size):
  in_size = int(input.get_shape()[1])
  W = weight_variable([in_size, size])
  b = bias_variable([size])
  # tf.matmul is a matrix multiplication from tensorn. This is the basic idea of ML
  # multiply 2 matrix and add a bias. This is the foward when we implement ANN
  return tf.matmul(input, W) + b


def main():

  # shape=[None, 784] means We do not how many data, but each one is 784 size
  x = tf.placeholder(tf.float32, shape=[None, 784]) # for the image
  y_ = tf.placeholder(tf.float32, shape=[None, 10]) # in this case, we do know we will have 10 possible outputs for y_

  # reshape for 28x28, and only macro - 1 channel
  x_image = tf.reshape(x, [-1, 28, 28, 1])

  # create first convolutional followed by pooling
  conv1 = conv_layer(x_image, shape=[5, 5, 1, 32]) # in this case, a filter of 5x5, used 32 times over the image
  # the result, which is 28x28x32, we feed to pooling
  conv1_pool = max_pool_2x2(conv1) # the result of this first polling will be 14X14X32

  # create first convolutional followed by pooling
  conv2 = conv_layer(conv1_pool, shape=[5, 5, 32, 64]) # the result here will be 14X14X64
  conv2_pool = max_pool_2x2(conv2) # the result will be 7X7X64

  # flat the final results, for then put in a fully connected layer
  # since the result data is 7X7X64 and we want to flat, we need then 1024 - Just a big array
  conv2_flat = tf.reshape(conv2_pool, [-1, 7*7*64])

  # create fully connected layer and train - Foward
  full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))

  # for dropout
  keep_prob = tf.placeholder(tf.float32)
  full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob) # for test, we will use full drop(no drops)

  # for output - For training
  y_conv = full_layer(full1_drop, 10) # In this case, weights will have size of 10 - Because we have 10 classes as output


  ###### Training #####
  DATA_DIR = '/tmp/data'
  NUM_STEPS = 1000
  MINIBATCH_SIZE = 100
  STEPS = 5000
  
  # read data
  mnist = input_data.read_data_sets(DATA_DIR, one_hot=True)

  # create our entropy, for loss mesuare
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y_conv))


  train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
  correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(STEPS):
      batch = mnist.train.next_batch(50)
      if i % 100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x: batch[0],
        y_: batch[1],
        keep_prob: 1.0})
        print("step {}, training accuracy {}".format(i, train_accuracy))

      sess.run(train_step, feed_dict={x: batch[0], y_: batch[1],keep_prob: 0.5})

    X = mnist.test.images.reshape(10, 1000, 784)
    Y = mnist.test.labels.reshape(10, 1000, 10)
    test_accuracy = np.mean([sess.run(accuracy,
                            feed_dict={x:X[i], y_:Y[i],keep_prob:1.0})
                            for i in range(10)])

  print("test accuracy: {}".format(test_accuracy))



if __name__ == '__main__':
  main()



