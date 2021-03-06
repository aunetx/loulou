import numpy as np


# Relu
def relu(y) -> np.ndarray:
    return np.maximum(y, 0)


def relu_prime(y) -> np.ndarray:
    return y > 0


# Leaky relu
def leaky_relu(y) -> np.ndarray:
    return np.where(y > 0, y, y * 0.01)


def leaky_relu_prime(y) -> np.ndarray:
    return (y >= 0) + (y < 0)*0.01


# Linear
def linear(y) -> np.ndarray:
    return y


def linear_prime(y) -> np.ndarray:
    return 1


# Heavyside
def heaviside(y) -> np.ndarray:
    return 1 * (y > 0)


def heaviside_prime(y) -> np.ndarray:
    return 0


# Sigmoid
def sigmoid(y) -> np.ndarray:
    return 1 / (1 + np.exp(-y))


def sigmoid_prime(y) -> np.ndarray:
    return sigmoid(y) * (1 - sigmoid(y))


# Tanh
def tanh(y) -> np.ndarray:
    return np.tanh(y)


def tanh_prime(y) -> np.ndarray:
    return 1 - tanh(y)**2


# Arctan
def arctan(y) -> np.ndarray:
    return np.arctan(y)


def arctan_prime(y) -> np.ndarray:
    return 1 / (y**2 + 1)


# Softmax
def softmax(y) -> np.ndarray:
    e = np.exp(y - np.max(y))
    return e / e.sum(axis=0)


# TODO implement a solution to derive softmax : the result has not the same shape as the input
def softmax_prime(y) -> np.ndarray:
    s = y.reshape(-1, 1)
    return np.mean(np.diagflat(s) - np.dot(s, s.T))


def listToActivations(activations_list, architecture) -> list:
    activations_fn = []
    activations_prime = []
    for id, _ in enumerate(architecture):
        if id < len(activations_list):

            if activations_list[id] == 'relu':
                activations_fn.append(relu)
                activations_prime.append(relu_prime)

            elif activations_list[id] == 'leaky_relu':
                activations_fn.append(leaky_relu)
                activations_prime.append(leaky_relu_prime)

            elif activations_list[id] == 'linear':
                activations_fn.append(linear)
                activations_prime.append(linear_prime)

            elif activations_list[id] == 'heaviside':
                activations_fn.append(heaviside)
                activations_prime.append(heaviside_prime)

            elif activations_list[id] == 'sigmoid':
                activations_fn.append(sigmoid)
                activations_prime.append(sigmoid_prime)

            elif activations_list[id] == 'tanh':
                activations_fn.append(tanh)
                activations_prime.append(tanh_prime)

            elif activations_list[id] == 'arctan':
                activations_fn.append(arctan)
                activations_prime.append(arctan_prime)

            elif activations_list[id] == 'softmax':
                print(
                    'Error : transfert function `', activations_list[id], '` is not fully implemented.', sep='')
                exit(1)
                activations_fn.append(softmax)
                activations_prime.append(softmax_prime)

            else:
                print(
                    'Error : transfert function `', activations_list[id], '` does not exist.', sep='')
                exit(1)

        # If not defined, fallback function is relu
        else:
            activations_fn.append(relu)
            activations_prime.append(relu_prime)

    return activations_fn, activations_prime
