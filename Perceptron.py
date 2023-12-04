import numpy as np

class Perceptron():
    def __init__(self, num_features):
        self.weights = np.random.randn(num_features)
        self.bias = 0.0

    def forward_pass(self, x):
        """

        Args:
            X (numpy ndarray): input features

        Returns:
             (float): output

        """
        z = np.dot(x, self.weights) + self.bias
        a = np.where( z >= 0, 1, 0)
        return a
    def train(self, X, y, learning_rate=0.05, epochs=100):

        self.errors_ = []

        for epoch in range(epochs):
            errors = 0
            for x_i, y_i in zip(X, y):
                yhat_i = self.forward_pass(x_i)
                w_delta = learning_rate * (y_i - yhat_i) * x_i
                b_delta = learning_rate * (y_i - yhat_i)
                self.weights = self.weights + w_delta
                self.bias = self.bias + b_delta
                errors += int(learning_rate * (y_i - yhat_i) != 0.0)
            self.errors_.append(errors)
        return self


