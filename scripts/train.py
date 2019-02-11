from loulou import *
import json
import sys
import argparse

if __name__ == '__main__':

    params = {}
    params['epochs'] = 15
    params['batch'] = 20
    params['learning_rate'] = 0.03
    params['save_timeout'] = 5
    params['reduce_output'] = 0
    architecture = [(784, 10)]
    filename = None

    #   Parser
    parser = argparse.ArgumentParser(description='Train a loulou-based neural network.')
    parser.add_argument('-f', '--filename', dest='filename', type=str,
        help='name of the file to write, extension added automatically (default : none)') #    Need to set option for other paths
    parser.add_argument('-e', '--epochs', dest='epochs', type=int,
        help='number of epochs (default : 15, -1 for infinity)')
    parser.add_argument('-b', '--batch-size', dest='batch_size', type=int,
        help='size of batch (default : 20)')
    parser.add_argument('-l', '--learning-rate', dest='learning_rate', type=float,
        help='learning rate of the network (default : 0.03)')
    parser.add_argument('-s', '--save-timeout', dest='save_timeout', type=int,
        help='number N of epochs to save automatically the file as filename_epoch_N (default : 0, disabled)')
    parser.add_argument('-a', '--architecture', nargs='+', type=int,
        help='architecture of the hidden layers (default : none)')
    parser.add_argument('-r', '--reduce-output', dest='reduce_output', action="count",
        help='reduce verbosity of output (you can type several)')
    args = parser.parse_args()

    if args.epochs is not None:
        params['epochs'] = args.epochs
    if args.batch_size is not None:
        params['batch'] = args.batch_size
    if args.learning_rate is not None:
        params['learning_rate'] = args.learning_rate
    if args.architecture is not None:
        architecture = listToArch(args.architecture)
    if args.filename is not None:
        filename = args.filename
    if args.save_timeout is not None:
        params['save_timeout'] = args.save_timeout
    if args.reduce_output is not None:
        params['reduce_output'] = args.reduce_output

    params = json.dumps(params)
    accuracy = runTrain(params, architecture, file=filename)
