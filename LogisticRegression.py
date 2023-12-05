import numpy as np

class LogisticRegression():
    """
    Simple logistic regression model implemented from scratch using numpy

    Args:
     - num_features (int): The number of features of the input

    Attributes:
     - weights (numpy ndarray): Array containing the weights of the model
     - bias (float): Bias term of the model

    """
    def __init__(self, num_features):
        self.weights = np.random.randn(num_features)
        self.bias = 0.0

    def forward_pass(self, x):
        """
        Forward pass of the model - computes the output

        Args:
         - X (numpy ndarray): Matrix containing the features of the data points
         - y (numpy ndarray): Array containing the true labels of the data points

        Returns:
         - a (float): output: Scalar value over (0, 1)

        """
        z = np.dot(x, self.weights) + self.bias
        a = 1/(1+np.exp(-z))
        return a

    def compute_loss(self, x, y):
        a = self.forward_pass(x)
        loss = -(y*np.log(a) + (1-y)*np.log(1-a))
        return loss

    def compute_gradient(self, x, y, a):
        dj_dw = (a - y)*x
        dj_db = a - y
        return dj_dw, dj_db

    def train(self, X, y, learning_rate=0.05, epochs=100):

        for epoch in range(epochs):
            for x_i, y_i in zip(X, y):
                a_i = self.forward_pass(x_i)
                dj_dw, dj_db = self.compute_gradient(x_i, y_i, a_i)
                self.weights = self.weights - learning_rate*dj_dw
                self.bias = self.bias - learning_rate*dj_db

        return self

    def predict(self, X):
        y_hat = []
        for xi in X:
            a = self.forward_pass(xi)
            y_hat_i = np.where(a >= 0.5, 1, 0)
            y_hat.append(y_hat_i.item())
        return y_hat


