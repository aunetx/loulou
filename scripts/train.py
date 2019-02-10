import mnist
from ffl import *

if __name__ == '__main__':
    filename = 'four layers.npy'
    filename = '../trains/' + filename
    trX, trY, teX, teY = mnist.load_data()
    weights = [np.random.randn(*w) * 0.1 for w in [(784, 400), (400,200), (200,100), (100, 10)]]
    train(weights, trX, trY, teX, teY, filename=filename, epochs=0)
