import numpy as np
import json
import os
import mnist

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights) # |grads| : weights corrections matrix
    a = feed_forward(X, weights) # Feeding network and storing values of layers in |a|
    delta = a[-1] - Y # |delta| : global network error (here)
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1): # Looping backward
        delta = (a[i] > 0) * delta.dot(weights[i].T) # |delta| : error of the layer (here)
        grads[i-1] = a[i-1].T.dot(delta) # calculating errors of weights and storing onto |grads|
    return grads / len(X)

def train(weights, trX, trY, teX, teY, filename, epochs, batch, learning_rate):
    accuracy = {}
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    accuracy[0] = np.mean(prediction == np.argmax(teY, axis=1))
    print(0, accuracy[0])
    for i in range(epochs):
        for j in range(0, len(trX), batch):
            X, Y = trX[j:j+batch], trY[j:j+batch]
            weights -= learning_rate * grads(X, Y, weights)
        prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
        accuracy[i+1] = np.mean(prediction == np.argmax(teY, axis=1))
        print(i+1, accuracy[i+1])
    if filename:
        path = os.path.dirname(__file__)
        filename = os.path.join(path, filename)
        save(weights, filename)
    return accuracy

def save(weights, filename):
    np.save(filename,weights)
    print('Data saved successfully into ',filename)

def convertJson(pred):
    out = {}
    out['hot_prediction'] = list(pred)
    out['prediction'] = int(np.argmax(pred))
    return json.dumps(out)

def runTrain(params, architecture, file='trained.npy'):
    params = json.loads(params)
    epochs = params['epochs']
    batch = params['batch']
    learning_rate = params['learning_rate']
    file = '../trains/' + file
    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in architecture]
    return train(weights, trX, trY, teX, teY, file, epochs, batch, learning_rate)

def listToArch(list):
    arch = []
    id = 0
    for hl in list:
        if id == 0:
            arch.append((784,hl))
            if id == len(list) - 1:
                arch.append((hl,10))
        elif id == len(list) - 1:
            arch.append((lastHl, hl))
            arch.append((hl,10))
        else:
            arch.append((lastHl, hl))
        lastHl = hl
        id += 1
    return arch
