from tqdm import tqdm
import numpy as np
import json
import sys
import os

import activations
import mnist
import utils


def feed_forward(X_input: np.ndarray, weights: list, activation_fn: list) -> np.ndarray:
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


def grads(x: np.ndarray, y_expected: np.ndarray, weights: list, activations_fn: list, activations_prime: list) -> np.ndarray:
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
    grads: np.ndarray = np.empty_like(weights)
    grads[-1] = y[-2].T.dot(delta)

    # Backward loop
    for i in range(len(y)-2, 0, -1):

        # Calculate error of each layer
        delta = delta.dot(weights[i].T) * activations_prime[i](y[i])

        # Calculate errors of weights
        grads[i-1] = y[i-1].T.dot(delta)

    return grads / len(x)


def train(weights: list, trX: np.ndarray, trY: np.ndarray, teX: np.ndarray, teY: np.ndarray, activations_fn: list, activations_prime: list, filename: np.ndarray, epochs: int, batch: int, learning_rate: float, save_timeout: int, reduce_output: int) -> dict:
    path = os.path.dirname(__file__)
    accuracy = []

    # Make prediction with the untrained network
    prediction: np.ndarray = np.argmax(feed_forward(
        teX, weights, activations_fn)[-1], axis=1)
    accuracy.append(np.mean(prediction == np.argmax(teY, axis=1)))

    if reduce_output <= 1:
        print('Accuracy of epoch 0 :', accuracy[0])
    elif reduce_output == 2:
        print(0, accuracy[0])

    if epochs < 0:
        epochs = 99999999999

    # Epochs loop
    for i in range(epochs):
        if reduce_output < 1:

            pbar = tqdm(range(0, len(trX), batch))
        else:
            pbar = range(0, len(trX), batch)

        # Batches loop
        for j in pbar:
            if reduce_output < 1:
                pbar.set_description("Processing epoch %s" % (i+1))

            # Select training data
            X, Y = trX[j:j+batch], trY[j:j+batch]

            # Correct the network
            weights -= learning_rate * \
                grads(X, Y, weights, activations_fn, activations_prime)

        # Make prediction for epoch
        prediction = np.argmax(feed_forward(
            teX, weights, activations_fn)[-1], axis=1)
        accuracy.append(np.mean(prediction == np.argmax(teY, axis=1)))

        if reduce_output < 2:
            print('Accuracy of epoch', i+1, ':', accuracy[i+1])
        if reduce_output == 2:
            print(i+1, accuracy[i+1])

        # Save temp file if set so
        if filename:
            if save_timeout > 0:
                if i % save_timeout == 0:
                    temp_filename = '../trains/temp/' + \
                        filename + '_epoch_' + str(i) + '.npz'
                    temp_filename = os.path.join(path, temp_filename)
                    utils.save(weights, activations_fn,
                               temp_filename, reduce_output)

    # Save final file
    if filename:
        filename = os.path.join(path, '../trains/' + filename + '.npz')
        utils.save(weights, activations_fn, filename, reduce_output)

    return accuracy


def runTrain(params: dict, architecture: list, file=None) -> dict:
    params: dict = json.loads(params)
    epochs: int = params['epochs']
    batch: int = params['batch']
    learning_rate: float = params['learning_rate']
    save_timeout: int = params['save_timeout']
    reduce_output: int = params['reduce_output']
    activations_arch, primes_arch = activations.listToActivations(
        params['activations'], architecture)

    # Print network visualization
    if reduce_output < 1:
        utils.print_network_visualization(
            architecture, activations_arch, epochs, batch, learning_rate)

    # Load data
    #   TODO do not load arbitrary data
    trX, trY, teX, teY = mnist.load_data()

    # Init weights
    weights = [np.random.randn(*w) * 0.1 for w in architecture]

    # Train network
    return train(weights, trX, trY, teX, teY, activations_arch, primes_arch, file, epochs, batch, learning_rate, save_timeout, reduce_output)
