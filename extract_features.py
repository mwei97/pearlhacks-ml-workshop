from __future__ import print_function
import os
import numpy as np
from PIL import Image

DATA_PATH = "./data"
TRAINING_B_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "bird.train")
VALIDATION_B_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "bird.val")
TESTING_B_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "bird.test")
TRAINING_P_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "plane.train")
VALIDATION_P_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "plane.val")
TESTING_P_IMAGE_LIST_PATH = os.path.join(DATA_PATH, "plane.test")

#
# Generates an 8x8x8 color histogram for image img. Greyscale
# images are handled by duplicating single channel 3 times
#
def generateHistogram(image_path):
    im = Image.open(os.path.join(DATA_PATH, image_path))
    pixels = im.load()
    width, height = im.size

    # Initialize each color hist with 8 bins
    # spaced evenly across 0-255
    rhist = [0] * 8
    ghist = [0] * 8
    bhist = [0] * 8

    hist = [0] * 8 * 8 * 8

    for row in range(height):
        for col in range(width):
            value = pixels[col,row]

            r = value[0]
            g = value[1]
            b = value[2]

            rindex = int(r/32)
            gindex = int(g/32)
            bindex = int(b/32)

            hist[rindex * 1 + gindex * 8 + bindex * 64] += 1

    del im
    return np.asarray(hist)

if __name__ == "__main__":
    print("Extracting color histogram features for images...")
    
    training_image_paths = []
    validation_image_paths = []
    testing_image_paths = []

    with open(TRAINING_B_IMAGE_LIST_PATH, "r") as f:
        training_image_paths += filter(None, f.read().split("\n"))

    with open(TRAINING_P_IMAGE_LIST_PATH, "r") as f:
        training_image_paths += filter(None, f.read().split("\n"))

    with open(VALIDATION_B_IMAGE_LIST_PATH, "r") as f:
        validation_image_paths += filter(None, f.read().split("\n"))

    with open(VALIDATION_P_IMAGE_LIST_PATH, "r") as f:
        validation_image_paths += filter(None, f.read().split("\n"))

    with open(TESTING_B_IMAGE_LIST_PATH, "r") as f:
        testing_image_paths += filter(None, f.read().split("\n"))

    with open(TESTING_P_IMAGE_LIST_PATH, "r") as f:
        testing_image_paths += filter(None, f.read().split("\n"))

    index = 0
    total_images = len(training_image_paths) + len(validation_image_paths) + len(testing_image_paths)
    training_image_histograms = []
    validation_image_histograms = []
    testing_image_histograms = []
    for path in training_image_paths:
        print("%0.2f percent done, on %s" % (float(index) * 100 / total_images, path))
        index += 1
        training_image_histograms.append(generateHistogram(path))

    for path in validation_image_paths:
        print("%0.2f percent done, on %s" % (float(index) * 100 / total_images, path))
        index += 1
        validation_image_histograms.append(generateHistogram(path))

    for path in testing_image_paths:
        print("%0.2f percent done, on %s" % (float(index) * 100 / total_images, path))
        index += 1
        testing_image_histograms.append(generateHistogram(path))

    output_training = np.asarray(training_image_histograms)
    output_validation = np.asarray(validation_image_histograms)
    output_testing = np.asarray(testing_image_histograms)

    np.save("training_features.npy", output_training)
    np.save("validation_features.npy", output_validation)
    np.save("testing_features.npy", output_testing)   
