import numpy as np
import json
import os
import mnist
import sys
from tqdm import tqdm

import activations


def feed_forward(X_input, weights, activation_fn):
    """Feed fordward the network

    X_input     => input layer
    weights     => weights of every layer

    x           => data propagated in previous layers
    w           => weight of working layer
    z           => weighted propagation
    y           => activated propagation"""

    x = [X_input]

    for id, w in enumerate(weights):
        # Weighted average `z = w^T · x`
        z = x[-1].dot(w)
        # Activation function `y = g(x)`
        y = activation_fn[id](z)
        # Append `y` to previous layers
        x.append(y)

    return x


def grads(X, Y, weights, activation_fn):
    # |grads| : weights corrections matrix
    grads = np.empty_like(weights)

    # Feeding network and storing values of layers in |a|
    a = feed_forward(X, weights, activation_fn)

    # |delta| : global network error (here)
    delta = a[-1] - Y

    grads[-1] = a[-2].T.dot(delta)

    # Backward loop
    for i in range(len(a)-2, 0, -1):

        # |delta| : error of the layer (here)
        delta = (a[i] > 0) * delta.dot(weights[i].T)

        # calculating errors of weights and storing onto |grads|
        grads[i-1] = a[i-1].T.dot(delta)

    return grads / len(X)


def train(weights, trX, trY, teX, teY, filename, epochs, batch, learning_rate, save_timeout, reduce_output, activation_fn):
    path = os.path.dirname(__file__)
    accuracy = {}
    prediction = np.argmax(feed_forward(
        teX, weights, activation_fn)[-1], axis=1)
    accuracy[0] = np.mean(prediction == np.argmax(teY, axis=1))
    if reduce_output < 2:
        print('Accuracy of epoch', 0, ':', accuracy[0])
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
                pbar.set_description("Processing epoch %s" % (i+1))

            X, Y = trX[j:j+batch], trY[j:j+batch]
            weights -= learning_rate * grads(X, Y, weights, activation_fn)
        prediction = np.argmax(feed_forward(
            teX, weights, activation_fn)[-1], axis=1)
        accuracy[i+1] = np.mean(prediction == np.argmax(teY, axis=1))
        if reduce_output < 2:
            print('Accuracy of epoch', i+1, ':', accuracy[i+1])
        if reduce_output == 2:
            print(i+1, accuracy[i+1])
        if filename:
            if save_timeout > 0:
                if i % save_timeout == 0:
                    temp_filename = '../trains/temp/' + \
                        filename + '_epoch_' + str(i) + '.npy'
                    temp_filename = os.path.join(path, temp_filename)
                    save(weights, temp_filename, reduce_output)
    if filename:
        filename = os.path.join(path, '../trains/' + filename + '.npy')
        save(weights, filename, reduce_output)
    return accuracy


def save(weights, filename, reduce_output):
    np.save(filename, weights)
    if reduce_output < 2:
        print('Data saved successfully into ', filename)


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
    activations_arch, primes_arch = listToActivations(
        params['activations'], architecture)

    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in architecture]
    return train(weights, trX, trY, teX, teY, file, epochs, batch, learning_rate, save_timeout, reduce_output, activations_arch)


def listToArch(list):
    arch = []
    id = 0
    for hl in list:
        if id == 0:
            arch.append((784, hl))
            if id == len(list) - 1:
                arch.append((hl, 10))
        elif id == len(list) - 1:
            arch.append((lastHl, hl))
            arch.append((hl, 10))
        else:
            arch.append((lastHl, hl))
        lastHl = hl
        id += 1
    return arch


def listToActivations(activations_list, architecture):
    activations_fn = []
    activations_prime = []
    for id, _ in enumerate(architecture):
        if id < len(activations_list):

            if activations_list[id] == 'relu':
                print(
                    'Activation `relu` successfully used for layer', id)
                activations_fn.append(activations.relu)
                activations_prime.append(activations.relu_prime)

            elif activations_list[id] == 'sigmoid':
                print(
                    'Sigmoid not defined, `relu` used for layer', id)
                activations_fn.append(activations.relu)
                activations_prime.append(activations.relu_prime)

            else:
                print(
                    'Error :', activations_list[id], 'not defined as activation function yet.')
                exit(1)

        # If not defined, fallback function is relu
        else:
            print(
                'Activation not defined, `relu` used for layer', id)
            activations_fn.append(activations.relu)
            activations_prime.append(activations.relu_prime)

    return activations_fn, activations_prime
