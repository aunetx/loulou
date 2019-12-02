import numpy as np
import json
import os
import mnist
import sys
from tqdm import tqdm

import activations
import utils


def feed_forward(X_input, weights, activation_fn):
    """Feed fordward the network

    X_input     => input layer
    weights     => weights of every layer

    x           => data propagated in previous layers
    w           => weight of working layer
    z           => weighted propagation
    y           => activated propagation"""

    x = [X_input]

    # Forward loop
    for id, w in enumerate(weights):
        # Weighted average `z = w^T · x`
        z = x[-1].dot(w)
        # Activation function `y = g(x)`
        y = activation_fn[id](z)
        # Append `y` to previous layers
        x.append(y)

    return x


def grads(x, y_expected, weights, activations_fn, activations_prime):
    """Calculate errors corrections with backward propagation

    x           => input layer
    y_expected  => expected output layer
    weights     => weights of every layer

    y           => actual output
    delta       => global (output) error of network
    grads       => gradient (correction) of weights

    """

    # Forward propagation to catch network datas
    y = feed_forward(x, weights, activations_fn)

    # Calculate global error
    delta = y[-1] - y_expected

    # Calculate error of output weights layer
    grads = np.empty_like(weights)
    grads[-1] = y[-2].T.dot(delta)

    # Backward loop
    for i in range(len(y)-2, 0, -1):

        # Calculate error of each layer
        delta = delta.dot(weights[i].T) * activations_prime[i](y[i])

        # Calculate errors of weights
        grads[i-1] = y[i-1].T.dot(delta)

    return grads / len(x)


def train(weights, trX, trY, teX, teY, filename, epochs, batch, learning_rate, save_timeout, reduce_output, activations_fn, activations_prime):
    path = os.path.dirname(__file__)
    accuracy = {}
    prediction = np.argmax(feed_forward(
        teX, weights, activations_fn)[-1], axis=1)
    accuracy[0] = np.mean(prediction == np.argmax(teY, axis=1))
    if reduce_output < 2:
        print('Accuracy of epoch 0 :', accuracy[0])
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
            weights -= learning_rate * \
                grads(X, Y, weights, activations_fn, activations_prime)
        prediction = np.argmax(feed_forward(
            teX, weights, activations_fn)[-1], axis=1)
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
                    utils.save(weights, temp_filename, reduce_output)
    if filename:
        filename = os.path.join(path, '../trains/' + filename + '.npy')
        utils.save(weights, filename, reduce_output)
    return accuracy


def runTrain(params, architecture, file=None):
    params = json.loads(params)
    epochs = params['epochs']
    batch = params['batch']
    learning_rate = params['learning_rate']
    save_timeout = params['save_timeout']
    reduce_output = params['reduce_output']
    activations_arch, primes_arch = activations.listToActivations(
        params['activations'], architecture)

    if reduce_output < 1:
        print_network_visualization(
            architecture, activations_arch, epochs, batch, learning_rate)

    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in architecture]
    return train(weights, trX, trY, teX, teY, file, epochs, batch, learning_rate, save_timeout, reduce_output, activations_arch, primes_arch)


def print_network_visualization(architecture, activations_arch, epochs, batch, learning_rate):
    print('Network has', len(architecture) - 1, 'hidden layers :')

    print('     layer [0]  -->  784   neurons, inputs')

    for id, layer in enumerate(architecture):
        if id != 0:
            print('     layer ['+str(id)+']  -->  ' +
                  str(layer[0]) + (' '*(6-len(str(layer[0])))) + 'neurons,', activations_arch[id-1].__name__)

    print('     layer ['+str(id+1)+']  -->  10    neurons,',
          activations_arch[id].__name__, '')

    print('Hyperparameters : epochs = ', epochs, ', batches = ',
          batch, ', learning rate = ', learning_rate, '\n', sep='')
