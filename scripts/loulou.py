import mnist
from rnn import *

if __name__ == '__main__':
    filename = 'four layers.npy'
    filename = 'blended/' + filename
    trX, trY, teX, teY = mnist.load_data()
