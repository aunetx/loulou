from ffl import *
import json

if __name__ == '__main__':
    params = {}
    params['epochs'] = 2
    params['batch'] = 20
    params['learning_rate'] = 0.03
    params = json.dumps(params)

    architecture = [(784, 100), (100, 10)]

    filename = 'trained.npy'
    accuracy = runTrain(params, architecture, file=filename)
