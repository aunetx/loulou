import mnist
from ffl import *
import json

def runTrain(params, file='trained.npy'):
    params = json.loads(params)
    epochs = params['epochs']
    batch = params['batch']
    learning_rate = params['learning_rate']
    file = '../trains/' + file
    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in [(784, 400), (400,200), (200,100), (100, 10)]]
    train(weights, trX, trY, teX, teY, 'trained.npy', epochs, batch, learning_rate)

if __name__ == '__main__':
    params = {}
    params['epochs'] = 30
    params['batch'] = 20
    params['learning_rate'] = 0.03
    params = json.dumps(params)

    filename = 'trained.npy'
    runTrain(params, file=filename)
