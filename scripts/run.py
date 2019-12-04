#!/usr/bin/env python3

import matplotlib.image as image
import numpy as np
import argparse
import sys

from loulou import feed_forward
from utils import convertJson
import activations

if __name__ == '__main__':
    verbosity = 0

    parser = argparse.ArgumentParser(
        description='Utility to run a loulou-based neural network.')
    parser.add_argument('-f', '--file', dest='file', type=str, required=True,
                        help='path to the `.npz` training file')
    parser.add_argument('-i', '--image', dest='image', type=str, required=True,
                        help='path to the image to predict')
    parser.add_argument('-v', '--verbosity', dest='verbosity', action="count",
                        help='add verbosity to the output (you can type several)')
    parser.add_argument('-s', '--show-image', dest='show_image', action="store_true",
                        help='show the image in matplotlib before prediction')
    parser.add_argument('-j', '--return-json', dest='return_json', action="store_true",
                        help='print the prediction and hot vector in a json format')
    args = parser.parse_args()

    if args.verbosity is not None:
        verbosity = args.verbosity

    # Load weights and activations from file
    try:
        # TODO change behavior to allow_pickle=False for security
        file = np.load(args.file, allow_pickle=True)
        weights = file['weights']
        activations_names = file['activations']

        # Show optionnal informations if contained in file
        if verbosity > 0:
            try:
                infos = list(file['optionnal_infos'])
                try:
                    print('Informations of training : accuracy =', infos[0], ' learning rate =', infos[1],
                          ' epochs =', infos[2], ' batches=', infos[3])
                except IndexError:
                    print('Error loading optionnal datas.')
            except KeyError:
                print('Optionnal datas not provided in the file.')

        activations_fn = activations.listToActivations(
            activations_names, weights)[0]
        if verbosity > 0:
            print('Activations used :', activations_names)

    except FileNotFoundError:
        print(
            'Error ! File ['+args.file+'] could not be opened, please check that it exists.')
        exit()

    # Load image data
    try:
        img = image.imread(args.image)

        # Show image if asked for
        if args.show_image:
            import matplotlib.pyplot as plt
            imgplot = plt.imshow(img)
            plt.show()

    except FileNotFoundError:
        print(
            'Error ! Image ['+args.image+'] could not be opened, please check that it exists.')
        exit()

    # Convert image to readable matrix
    topred = 1 - img.reshape(784, 4).mean(axis=1)
    # Make the prediction
    prediction = feed_forward(topred, weights, activations_fn)[-1]
    # Print final output
    if args.return_json:
        print(convertJson(prediction))
    else:
        if verbosity > 0:
            print('Hot ones vector :', list(prediction))
            print('Final prediction :', prediction.argmax())
        else:
            print(prediction.argmax())
