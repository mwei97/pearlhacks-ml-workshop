# Pearl Hacks 2017 Intro to ML demo
#
# runs through a 2-class classification
# using an SVM with a linear kernel
# adapted from the sklearn example code at
# http://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html

from __future__ import print_function
import os
import numpy as np
from PIL import Image
from sklearn import svm, metrics

FEATURE_DIR = "./color_histogram_features"
TRAINING_FEATURES_FILE = os.path.join(FEATURE_DIR, "training_features.npy")
VALIDATION_FEATURES_FILE = os.path.join(FEATURE_DIR, "validation_features.npy")
TESTING_FEATURES_FILE = os.path.join(FEATURE_DIR, "testing_features.npy")
BIRD_LABEL = 0
PLANE_LABEL = 1

if __name__ == "__main__":

    # read extracted features from numpy files
    training_data = np.load(TRAINING_FEATURES_FILE)
    validation_data = np.load(VALIDATION_FEATURES_FILE)
    testing_data = np.load(TESTING_FEATURES_FILE)

    # assign corresponding labels for the images
    # (e.g., for the training images, we had 1500
    # bird images prepended to 1500 plane images when
    # we saved the .npy features in extract_features.py)
    training_labels = np.asarray([BIRD_LABEL] * 1500 + [PLANE_LABEL] * 1500)
    validation_labels = np.asarray([BIRD_LABEL] * 100 + [PLANE_LABEL] * 100)
    testing_labels = np.asarray([BIRD_LABEL] * 400 + [PLANE_LABEL] * 400)

    # train the default sklearn linear SVM
    classifier = svm.LinearSVC()
    classifier.fit(training_data, training_labels)

    # test the trained SVM
    predictions = classifier.predict(testing_data)

    # show results on the test data
    print("Classification stats for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(testing_labels, predictions)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testing_labels, predictions))
