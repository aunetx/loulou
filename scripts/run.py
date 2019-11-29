import numpy as np
#from ffl import *
import sys
import matplotlib.image as image
from loulou import feed_forward, convertJson

if __name__ == '__main__':
    #   Handling errors for bad arguments
    try:
        assert len(sys.argv) == 3
    except AssertionError:
        print("Error ! Please give two arguments : path to weights file and to image to predict.")
        exit()

    #   Loading weights matrix
    filename = sys.argv[1]
    try:
        weights = np.load(filename)
    except FileNotFoundError:
        print(
            "Error ! Weights matrix file could not be opened, please check that it exists.")
        print("Fichier : ", filename)
        exit()

    #   Loading image data
    img = sys.argv[2]
    try:
        img = image.imread(img)
    except FileNotFoundError:
        print("Error ! Image could not be opened, please check that it exists.")
        print("Image : ", img)
        exit()

    #   Shaping image onto matrix
    topred = 1 - img.reshape(784, 4).mean(axis=1)
    #   Making prediction
    prediction = feed_forward(topred, weights)[-1]
    #   Printing json output
    print(convertJson(prediction))
