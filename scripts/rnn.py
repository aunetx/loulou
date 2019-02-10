import numpy as np

def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights):
    grads = np.empty_like(weights) # |grads| : weights corrections matrix
    a = feed_forward(X, weights) # Feeding network and storing values of layers in |a|
    delta = a[-1] - Y # |delta| : global network error (here)
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1): # Looping backward
        delta = (a[i] > 0) * delta.dot(weights[i].T) # |delta| : error of the layer (here)
        grads[i-1] = a[i-1].T.dot(delta) # calculating errors of weights and storing onto |grads|
    return grads / len(X)

def train(weights, trX, trY, teX, teY, filename=False, epochs=30, batch=20, learning_rate=0.03):
    prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
    print(0, np.mean(prediction == np.argmax(teY, axis=1)))
    for i in range(epochs):
        for j in range(0, len(trX), batch):
            X, Y = trX[j:j+batch], trY[j:j+batch]
            weights -= learning_rate * grads(X, Y, weights)
        prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
        print(i+1, np.mean(prediction == np.argmax(teY, axis=1)))
    if filename:
        save(weights, filename)

def save(weights, filename):
    np.save(filename,weights)
    print('Data saved successfully into ',filename)
