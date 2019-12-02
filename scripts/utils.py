import numpy as np
import json


def save(weights, filename, reduce_output):
    np.save(filename, weights)
    if reduce_output < 2:
        print('Data saved successfully into ', filename)


def convertJson(pred):
    out = {}
    out['hot_prediction'] = list(pred)
    out['prediction'] = int(np.argmax(pred))
    return json.dumps(out)


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
