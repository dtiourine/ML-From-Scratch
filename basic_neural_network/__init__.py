import numpy as np
def init_params():
    W1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5
    W2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2

def ReLU(z):
    return np.maximum(0, z)

def softmax(z):
    return np.exp(z) / np.sum(np.exp(z))
def forward_propogation(X, W1, b1, W2, b2):
    Z1 = np.dot(X, W1) + b1
    A1 = ReLU(Z1)
    Z2 = np.dot(A1, W2) + b2
    A2 = ReLU(Z2)
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y
def backward_propogation(Z1, A1, Z2, A2, W2, Y):
    one_hot_y = one_hot(Y)
    




