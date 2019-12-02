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
