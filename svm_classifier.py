# Pearl Hacks 2017 Intro to ML demo
#
# runs through a 2-class classification
# using an SVM with a linear kernel
# adapted from 
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

if __name__ == "__main__":

    # read features from numpy files
    training_data = np.load(TRAINING_FEATURES_FILE)
    validation_data = np.load(VALIDATION_FEATURES_FILE)
    testing_data = np.load(TESTING_FEATURES_FILE)

    training_labels = np.asarray([1] * 1500 + [2] * 1500)
    validation_labels = np.asarray([1] * 100 + [2] * 100)
    testing_labels = np.asarray([1] * 400 + [2] * 400)

    #classifier = svm.SVC( gamma = 0.001 )
    classifier = svm.LinearSVC()
    classifier.fit(training_data, training_labels)

    predictions = classifier.predict(testing_data)

    print("Classification stats for classifier %s:\n%s\n"
          % (classifier, metrics.classification_report(testing_labels, predictions)))
    print("Confusion matrix:\n%s" % metrics.confusion_matrix(testing_labels, predictions))

 
