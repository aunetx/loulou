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

### Utilisation :
There are two ways to use loulou :
1. *Training a model :*\
  You need to define hyperparameters (number of `epochs`, `batch size` and `learning rate`)\
  You choose a name for the saved training (will be in `./trains/`)\
  And then you just run your training.\
  *Simple access support added soon*

2. *Making a prediction :*\
  This is as simple as :\
        ```
        python ./scripts/run.py [path/to/weights/file.npy] [path/to/image.png]
        ```  
  Or, for windows :\
        ```
        py .\scripts\run.py [path\to\weights\file.npy] [path\to\image.png]
        ```

## To go further :
### Versions :
* Version 1.0.0 : Initial version, works with another implementation - need some changes

* Version 1.1.0 : First working version, added two main scripts
