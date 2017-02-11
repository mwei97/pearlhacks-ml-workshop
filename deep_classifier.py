# Pearl Hacks 2017 Intro to ML demo
#
# runs through a tensorflow example for classifying
# between bird and plane images with a single layer
# adapted from the starter Tensorflow MNIST example
# at: https://www.tensorflow.org/tutorials/mnist/beginners/ 

from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

INPUT_SIZE = 512
NUMBER_OF_CLASSES = 2
FEATURE_DIR = "./color_histogram_features"
TRAINING_FEATURES_FILE = os.path.join(FEATURE_DIR, "training_features.npy")
VALIDATION_FEATURES_FILE = os.path.join(FEATURE_DIR, "validation_features.npy")
TESTING_FEATURES_FILE = os.path.join(FEATURE_DIR, "testing_features.npy")

def main(_):
 
    training_data = np.load(TRAINING_FEATURES_FILE)
    validation_data = np.load(VALIDATION_FEATURES_FILE)
    testing_data = np.load(TESTING_FEATURES_FILE)

    # assign corresponding labels for the images
    # (e.g., for the training images, we had 1500
    # bird images prepended to 1500 plane images when
    # we saved the .npy features in extract_features.py)
    training_labels, validation_labels, testing_labels = [], [], []
    for _ in range(1500):
        training_labels.append([1,0]) # bird labels
    for _ in range(1500):
        training_labels.append([0,1]) # plane labels
    for _ in range(100):
        validation_labels.append([1,0]) # bird labels
    for _ in range(100):
        validation_labels.append([0,1]) # plane labels
    for _ in range(400):
        testing_labels.append([1,0]) # bird labels
    for _ in range(400):
        testing_labels.append([0,1]) # plane labels
    training_labels = np.asarray(training_labels)
    validation_labels = np.asarray(validation_labels)
    testing_labels = np.asarray(testing_labels)

    x = tf.placeholder(tf.float32, [None, INPUT_SIZE])
    W = tf.Variable(tf.zeros([INPUT_SIZE, NUMBER_OF_CLASSES]))
    b = tf.Variable(tf.zeros([NUMBER_OF_CLASSES]))
    y = tf.matmul(x, W) + b

    y_ = tf.placeholder(tf.float32, [None, NUMBER_OF_CLASSES])

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits( labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    for _ in range(1000):
        sess.run(train_step, feed_dict={ x: training_data, y_: training_labels })

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print(sess.run(accuracy, feed_dict={x: testing_data, y_: testing_labels}))

if __name__ == "__main__":
    tf.app.run(main=main)
