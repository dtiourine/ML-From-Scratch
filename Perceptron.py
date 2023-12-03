import numpy as np

class Perceptron():
    def __init__(self, num_features):
        self.weights = np.random.randn(num_features)
        self.bias = 0.0

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
    def train(self, X, y, learning_rate=0.05, epochs=100):

        errors = []

        for epoch in range(epochs):
            errors = 0
            for xi, y in zip(X,y):
                yhat = self.forward_pass(xi)
                w_delta = self.learning_rate * (y - yhat) * xi
                b_delta = self.learning_rate * (y - yhat)
                self.weights = self.weights + w_delta
                self.bias = self.bias + b_delta
                errors += int(self.lr * (y - yhat) != 0.0)
            errors_.append(errors)
        return self


