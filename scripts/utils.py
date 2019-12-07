import numpy as np
import json


def save(weights: list, activations_fn_list: list, filename: str, no_infos: bool, infos: list, reduce_output: int) -> None:
    activations_names_list = []
    for act in activations_fn_list:
        activations_names_list.append(act.__name__)
    if not no_infos:
        np.savez_compressed(filename, weights=weights,
                            activations=activations_names_list, optionnal_infos=infos)
    else:
        np.savez_compressed(filename, weights=weights,
                            activations=activations_names_list)
    if reduce_output < 2:
        print('Data saved successfully into ', filename)


def convertJson(pred: np.ndarray) -> str:
    out = {}
    out['hot_prediction'] = list(pred)
    out['prediction'] = int(pred.argmax())
    return json.dumps(out)


def listToArch(list: list) -> list:
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


def print_network_visualization(architecture: list, activations_arch: list, epochs: int, batch: int, learning_rate: float) -> None:
    print('Network has', len(architecture) - 1, 'hidden layers :')

    print('     layer [0]  -->  784   neurons, inputs')

    for id, layer in enumerate(architecture):
        if id != 0:
            print('     layer ['+str(id)+']  -->  ' +
                  str(layer[0]) + (' '*(6-len(str(layer[0])))) + 'neurons,', activations_arch[id-1].__name__)

    print('     layer ['+str(id+1)+']  -->  10    neurons,',
          activations_arch[id].__name__, '')

    if epochs < 0:
        epochs = 'no limit'
    print('Hyperparameters : epochs = ', epochs, ', batches = ',
          batch, ', learning rate = ', learning_rate, '\n', sep='')
