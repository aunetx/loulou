import numpy as np
import sys
import matplotlib.image as image
import json

#   Erreur si le nombre d'arguments n'est pas bon.
try:
    assert len(sys.argv) == 3
except AssertionError:
    print("Erreur ! Veuillez donnez deux arguments, le nom du fichier de poids et de l'image à prédire.")
    exit()

#   On charge les poids
filename = sys.argv[1]
try:
    weights = np.load(filename)
except FileNotFoundError:
    print("Erreur ! Le fichier de poids n'a pas pu être ouvert, vérifiez qu'il existe bien.")
    print("Fichier : ",filename)
    exit()

#   On charge l'image
img = sys.argv[2]
try:
    img = image.imread(img)
except FileNotFoundError:
    print("Erreur ! L'image n'a pas pu être ouverte, vérifiez qu'elle existe bien.")
    print("Image : ",img)
    exit()

def predire(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

out = {}
topred = 1 - img.reshape(784,4).mean(axis=1)
prediction = predire(topred, weights)[-1]
out['accuracy'] = list(prediction)
out['prediction'] = int(np.argmax(prediction))
out_json = json.dumps(out)
print(out_json)
