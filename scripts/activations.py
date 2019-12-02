import numpy as np


# Relu
def relu(y):
    return np.maximum(y, 0)


def relu_prime(y):
    return y > 0


# Leaky relu
def leaky_relu(y):
    return np.where(y > 0, y, y * 0.01)


def leaky_relu_prime(y):
    return (y >= 0) + (y < 0)*0.01


# Linear
def linear(y):
    return y


def linear_prime(y):
    return 1


# Heavyside
def heaviside(y):
    return 1 * (y > 0)


def heaviside_prime(y):
    return 0


# Sigmoid
def sigmoid(y):
    return 1 / (1 + np.exp(-y))


def sigmoid_prime(y):
    return y * (1 - y)


# Tanh
def tanh(y):
    return np.tanh(y)


def tanh_prime(y):
    return 1 - y**2


# Arctan
def arctan(y):
    return np.arctan(y)


def arctan_prime(y):
    return 1 / y**2 + 1


def listToActivations(activations_list, architecture):
    activations_fn = []
    activations_prime = []
    for id, _ in enumerate(architecture):
        if id < len(activations_list):

            if activations_list[id] == 'relu':
                print(
                    'Activation `relu` successfully used for layer', id)
                activations_fn.append(relu)
                activations_prime.append(relu_prime)

            elif activations_list[id] == 'leaky_relu':
                print(
                    'Activation `leaky_relu` successfully used for layer', id)
                activations_fn.append(leaky_relu)
                activations_prime.append(leaky_relu_prime)

            elif activations_list[id] == 'linear':
                print(
                    'Activation `linear` successfully used for layer', id)
                activations_fn.append(linear)
                activations_prime.append(linear_prime)

            elif activations_list[id] == 'heaviside':
                print(
                    'Activation `heaviside` successfully used for layer', id)
                activations_fn.append(heaviside)
                activations_prime.append(heaviside_prime)

            elif activations_list[id] == 'sigmoid':
                print(
                    'Activation `sigmoid` successfully used for layer', id)
                activations_fn.append(sigmoid)
                activations_prime.append(sigmoid_prime)

            elif activations_list[id] == 'tanh':
                print(
                    'Activation `tanh` successfully used for layer', id)
                activations_fn.append(tanh)
                activations_prime.append(tanh_prime)

            elif activations_list[id] == 'arctan':
                print(
                    'Activation `arctan` successfully used for layer', id)
                activations_fn.append(arctan)
                activations_prime.append(arctan_prime)

            else:
                print(
                    'Error :', activations_list[id], 'not defined as activation function yet.')
                exit(1)

        # If not defined, fallback function is relu
        else:
            print(
                'Activation not defined, `relu` used for layer', id)
            activations_fn.append(relu)
            activations_prime.append(relu_prime)

    return activations_fn, activations_prime
