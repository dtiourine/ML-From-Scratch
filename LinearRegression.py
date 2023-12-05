import numpy as np

class LinearRegression():
    """
    A simple linear regression model implemented from scratch using numpy

    Args:
    - num_features (int): The number of weights in the model
    - epochs (int): The number of training steps/epochs

    Attributes:
    - bias: (float): The bias of the model
    - weights: (numpy.ndarray): The weights of the model

    """
    def __init__(self, num_features):
        """
       Initializes the weights and biases of the model

       Args:
        - num_weights (int): Defines the number of weights of the model

       Attributes:
        - bias (float): Bias term of the model
        - weights (numpy ndarray): Array containing the weights of the model

        """

        self.bias = 0.0
        self.weights = np.random.randn(num_features)

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

    def train(self, X, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            total_loss = 0
            for x_i, y_i in zip(X, y):
                y_hat_i = self.predict(x_i)
                loss = (y_hat_i - y_i) ** 2
                total_loss += loss
                dJ_dw = x_i * (y_hat_i - y_i)
                dJ_db = 2 * (y_hat_i - y_i)
                self.weights = self.weights - learning_rate * dJ_dw
                self.bias = self.bias - learning_rate * dJ_db
            print(f'[{epoch}] Avg Loss: {total_loss/len(y):.4f}')



