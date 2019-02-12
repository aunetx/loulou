import numpy as np
import json
import os
import mnist
import sys
from tqdm import tqdm

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

def train(weights, trX, trY, teX, teY, filename, epochs, batch, learning_rate, save_timeout, reduce_output):
    path = os.path.dirname(__file__)
    accuracy = {}
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    accuracy[0] = np.mean(prediction == np.argmax(teY, axis=1))
    if reduce_output < 2:
        print('Accuracy of epoch',0,':', accuracy[0])
    if reduce_output == 2:
        print(0, accuracy[0])
    if epochs < 0:
        epochs = 99999999999
    for i in range(epochs):
        if reduce_output < 1:
            pbar = tqdm(range(0, len(trX), batch))
        else:
            pbar = range(0, len(trX), batch)
        for j in pbar:
            if reduce_output < 1:
                pbar.set_description("Processing epoch %s" %(i+1))

            X, Y = trX[j:j+batch], trY[j:j+batch]
            weights -= learning_rate * grads(X, Y, weights)
        prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
        accuracy[i+1] = np.mean(prediction == np.argmax(teY, axis=1))
        if reduce_output < 2:
            print('Accuracy of epoch',i+1,':', accuracy[i+1])
        if reduce_output == 2:
            print(i+1, accuracy[i+1])
        if filename:
            if save_timeout > 0:
                if i % save_timeout == 0:
                    temp_filename = '../trains/temp/' + filename + '_epoch_' + str(i) + '.npy'
                    temp_filename = os.path.join(path, temp_filename)
                    save(weights, temp_filename, reduce_output)
    if filename:
        filename = os.path.join(path, '../trains/' + filename + '.npy')
        save(weights, filename, reduce_output)
    return accuracy

def save(weights, filename, reduce_output):
    np.save(filename,weights)
    if reduce_output < 2:
        print('Data saved successfully into ',filename)

def convertJson(pred):
    out = {}
    out['hot_prediction'] = list(pred)
    out['prediction'] = int(np.argmax(pred))
    return json.dumps(out)

def runTrain(params, architecture, file=None):
    params = json.loads(params)
    epochs = params['epochs']
    batch = params['batch']
    learning_rate = params['learning_rate']
    save_timeout = params['save_timeout']
    reduce_output = params['reduce_output']
    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in architecture]
    return train(weights, trX, trY, teX, teY, file, epochs, batch, learning_rate, save_timeout, reduce_output)

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
