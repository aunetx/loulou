import numpy as np
import json
import sys
import os

import activations
import mnist
import utils

# Stop on RuntimeWarning during matrix processing : prevent silent overflows
np.seterr(all='raise')


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
        # Weighted average `z = w · x`
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

    # Calculate the cost (average error)
    cost = 1/len(y) * np.sum((y[-1] - y_expected) ** 2)

    # Calculate error of output weights layer
    grads: np.ndarray = np.empty_like(weights)
    grads[-1] = y[-2].T.dot(delta)

    # Backward loop
    for i in range(len(y)-2, 0, -1):

        # Calculate error of each layer
        delta = delta.dot(weights[i].T) * activations_prime[i](y[i])

        # Calculate errors of weights
        grads[i-1] = y[i-1].T.dot(delta)

    return grads / len(x), cost


def train(weights: list, trainX: np.ndarray, trainY: np.ndarray, testX: np.ndarray, testY: np.ndarray, activations_fn: list, activations_prime: list, filename: np.ndarray, epochs: int, batch: int, learning_rate: float, save_timeout: int, graph: bool, no_infos: bool, reduce_output: int) -> dict:
    path = os.path.dirname(__file__)
    accuracy_table = []
    average_cost_table = []

    # Make prediction with the untrained network
    prediction = np.argmax(feed_forward(
        testX, weights, activations_fn)[-1], axis=1)
    accuracy = np.mean(prediction == np.argmax(testY, axis=1))
    accuracy_table.append(accuracy)

    initial_cost = 1/len(testY) * np.sum((prediction -
                                          np.argmax(testY, axis=1)) ** 2)
    average_cost_table.append(initial_cost)

    if reduce_output <= 1:
        print('Accuracy at epoch 0 :', accuracy, ' cost =', initial_cost)
    elif reduce_output == 2:
        print(0, accuracy, initial_cost)

    if epochs < 0:
        epochs = 99999999999

    # Epochs loop
    for i in range(epochs):
        cost_table = []

        if reduce_output < 1:
            try:
                from tqdm import tqdm
            except ImportError:
                print('Cannot find module `tqdm`!\nInstall it with `pip3 install tqdm` (or equivalent), or run the program with the argument `-r`.')
                exit(1)
            pbar = tqdm(range(0, len(trainX), batch))
        else:
            pbar = range(0, len(trainX), batch)

        # Batches loop
        for j in pbar:
            if reduce_output < 1:
                pbar.set_description("Processing epoch %s" % (i+1))

            # Select training data
            X, Y = trainX[j:j+batch], trainY[j:j+batch]

            # Correct the network
            grad, cost = grads(
                X, Y, weights, activations_fn, activations_prime)
            weights -= learning_rate * grad

            cost_table.append(cost)

        average_cost = np.mean(cost_table)
        average_cost_table.append(average_cost)

        # Make prediction for epoch
        prediction = np.argmax(feed_forward(
            testX, weights, activations_fn)[-1], axis=1)
        accuracy = np.mean(prediction == np.argmax(testY, axis=1))
        accuracy_table.append(accuracy)

        if reduce_output < 2:
            print('Accuracy at epoch', i+1, ':',
                  accuracy, ' cost =', average_cost)
        if reduce_output == 2:
            print(i+1, accuracy, average_cost)

        # Save temp file if set so
        if filename:
            if save_timeout > 0:
                if i % save_timeout == 0:
                    temp_filename = '../trains/temp/' + \
                        filename + '_epoch_' + str(i) + '.npz'
                    temp_filename = os.path.join(path, temp_filename)

                    infos = [accuracy, learning_rate, i, batch]

                    utils.save(weights, activations_fn,
                               temp_filename, no_infos, infos, reduce_output)

    # Show plot of accuracy and cost
    if graph:
        print('Plotting training evolution...', end=' ', flush=True)
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print('Cannot find module `matplotlib`!\nInstall it with `pip3 install matplotlib` (or equivalent), or run the program with the argument `-g`.')
            exit(1)
        plt.plot(range(1, epochs+1), average_cost_table, label='cost')
        plt.plot(range(0, epochs+1), accuracy_table, label='accuracy')
        plt.xlim(0, epochs)
        plt.ylim(0)
        plt.grid(axis='both', linestyle=':')
        plt.xlabel('Epoch number', fontsize=11)
        plt.legend()
        plt.show()
        plt.close()
        print('done !')

    # Save final file
    if filename:
        filename = os.path.join(path, '../trains/' + filename + '.npz')

        infos = [accuracy, learning_rate, epochs, batch]

        utils.save(weights, activations_fn, filename,
                   no_infos, infos, reduce_output)

    return (accuracy_table, average_cost_table), weights


def runTrain(params: dict, architecture: list, file=None) -> dict:
    params: dict = json.loads(params)
    epochs: int = params['epochs']
    batch: int = params['batch']
    learning_rate: float = params['learning_rate']
    save_timeout: int = params['save_timeout']
    graph: bool = params['graph']
    no_infos: bool = params['no_infos']
    reduce_output: int = params['reduce_output']
    activations_arch, primes_arch = activations.listToActivations(
        params['activations'], architecture)

    # Print network visualization
    if reduce_output < 1:
        utils.print_network_visualization(
            architecture, activations_arch, epochs, batch, learning_rate)

    # Load data
    trX, trY, teX, teY = mnist.load_data()

    # Init weights
    weights = [np.random.randn(*w) * 0.1 for w in architecture]

    # Train network
    tr, weights = train(weights, trX, trY, teX, teY, activations_arch, primes_arch, file,
                        epochs, batch, learning_rate, save_timeout, graph, no_infos, reduce_output)

    return tr
