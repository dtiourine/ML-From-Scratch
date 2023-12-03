import numpy as np

class Perceptron():
    def __init__(self, input_shape):
        self.bias = 0.0
        self.weights = np.random.randn(input_shape)

    def forward_pass(self, x):
        """

        Args:
            x (numpy ndarray): input features

        Returns:
             (float): output

        """
        z = np.dot(x, self.weights) + self.bias
        a = np.where( z >= 0, 1, 0)
        return a
    def train(self, x, y):

        yhat = self.forward_pass(x)

        w_delta = 
        b_delta =

        self.weights = self.weights + w_delta
        self.bias = self.bias + b_delta
