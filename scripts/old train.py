import numpy as np
import mnist as mnist

# Fordward propagation
def feed_forward(X, weights):
    a = [X]
    for w in weights:
        a.append(np.maximum(a[-1].dot(w),0))
    return a

def grads(X, Y, weights, square):
    grads = np.empty_like(weights) # |grads| : weights corrections matrix
    a = feed_forward(X, weights) # on nourrit le réseau et on stocke les valeurs des neurones dans 'a'
    if square:
        delta = a[-1]*a[-1] - Y*Y # on met l'erreur au carré
    else:
        delta = a[-1] - Y # on calcule l'erreur simple
    grads[-1] = a[-2].T.dot(delta)
    for i in range(len(a)-2, 0, -1):
        delta = (a[i] > 0) * delta.dot(weights[i].T)
        grads[i-1] = a[i-1].T.dot(delta)
    return grads / len(X)

def save(weights, filename):
    np.save(filename,weights)
    print('Data saved successfully into ',filename)

if __name__ == '__main__':
    learn = True
    save_it = True
    square = True
    filename = 'four layers.npy'
    filename = 'blended/' + filename
    trX, trY, teX, teY = mnist.load_data() # Load model
    if learn:
        weights = [np.random.randn(*w) * 0.1 for w in [(784, 400), (400,200), (200,100), (100, 10)]] # Initialiser les poids
        num_epochs, batch_size, learn_rate = 30, 20, 0.03 # Initialiser les hyperparamètres
        prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
        print(0, np.mean(prediction == np.argmax(teY, axis=1)))
        for i in range(num_epochs):
            for j in range(0, len(trX), batch_size):
                X, Y = trX[j:j+batch_size], trY[j:j+batch_size]
                weights -= learn_rate * grads(X, Y, weights, square)
            prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
            print(i+1, np.mean(prediction == np.argmax(teY, axis=1)))
        if save_it:
            save(weights,filename)

    if not learn:
        weights = np.load(filename)
        prediction = np.argmax(feed_forward(teX, weights)[-1], axis=1)
        print(np.mean(prediction == np.argmax(teY, axis=1)))
