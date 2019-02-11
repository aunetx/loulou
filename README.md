# Loulou

## Getting started
### Description :
This python projects aims to bring a new deep learning implementation to frond-end developers.\
It currently supports `MNIST` database and then can train neural networks on it, and predict grayscale 28*28px image that represents a number from 0 to 9.
### Prerequisites :
You must install `numpy` to use Loulou.\
In intend to show image that is predicted, you must also install `matplotlib`.

From `pip` :
```
pip install numpy
pip install matplotlib #(optional)
```
### Installation :
To install Loulou, just do :
```
git clone https://github.com/aunetx/loulou
cd loulou/
```

## Utilisation :
There are two ways to use loulou :
### *Training a model :*
To train a model, just do :
```
python ./scripts/train.py -e [number of epochs] -b [batch size] -l [learning rate] -a [architecture] -f [filename.npy]
```
Or, for windows :
```
py .\scripts\train.py -e [number of epochs] -b [batch size] -l [learning rate] -a [architecture] -f [filename.npy]
```
`architecture` is a list of digits that defines the number of neurons in each hidden layer.

For example : `python ./scripts/train.py -a 400 200 100` creates a network with 5 layers like :
1. 784 neurons - *Input layer*
2. 400 neurons - *First hidden layer*
3. 200 neurons - *Second hidden layer*
4. 100 neurons - *Third hidden layer*
5. 10 neurons - *Output layer*

All the arguments are optional : if you don't set a filename, the training is not saved.

### *Making a prediction :*
This is as simple as :
```
python ./scripts/run.py [path/to/weights/file.npy] [path/to/image.png]
```  
Or, for windows :
```
py .\scripts\run.py [path\to\weights\file.npy] [path\to\image.png]
```

## To go further :
### Version :
* Version 1.1.2 : output improvement for training added
