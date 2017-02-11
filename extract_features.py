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
# Generates an 8x8x8 color histogram for the image at image_path
#
def generateHistogram(image_path):
    im = Image.open(os.path.join(DATA_PATH, image_path))
    pixels = im.load()
    width, height = im.size

    # Initialize a color hist with 8x8x8 bins overall,
    # so each RGB color gets 8 bins
    hist = [0] * 8 * 8 * 8

    # Iterate over all pixels in the image, and
    # update each color frequency
    for row in range(height):
        for col in range(width):
            value = pixels[col,row]

            r = value[0]
            g = value[1]
            b = value[2]

            # Divide each color up into 8 bins (256/32=8)
            rindex = int(r/32)
            gindex = int(g/32)
            bindex = int(b/32)

            # Increment a bin in the 512 long hist based on
            # which bins we have. Each digit of the index
            # describes which RGB bin we want to increment
            hist[rindex * 1 + gindex * 8 + bindex * 64] += 1

    # Free the image
    del im

    # Turn the histogram into a numpy array
    # and normalize it according to the image size.
    # This is necessary since our input images are not
    # all the same size (bigger images would have
    # higher frequencies overall, which we don't want).
    return_hist = np.asarray(hist)
    return_hist = return_hist / (width * height)
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

    print("Saving features to '*_features.npy'...")
    np.save("training_features.npy", output_training)
    np.save("validation_features.npy", output_validation)
    np.save("testing_features.npy", output_testing)   
