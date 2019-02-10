import numpy as np
import sys
import matplotlib.image as image
import json

#   Prediction function (fordward propagation)
def predire(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

#   Json conversion function
def saveJson(pred):
    out = {}
    out['hot_prediction'] = list(pred)
    out['prediction'] = int(np.argmax(pred))
    return json.dumps(out)

#   Handling errors for arguments
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
    print("Error ! Weights matrix file could not be opened, please check that it exists.")
    print("Fichier : ",filename)
    exit()

#   Loading image data
img = sys.argv[2]
try:
    img = image.imread(img)
except FileNotFoundError:
    print("Error ! Image could not be opened, please check that it exists.")
    print("Image : ",img)
    exit()

#   Shaping image onto matrix
topred = 1 - img.reshape(784,4).mean(axis=1)
#   Making prediction
prediction = predire(topred, weights)[-1]
#   Printing json output
print(saveJson(prediction))
