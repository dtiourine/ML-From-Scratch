import numpy as np

class LinearRegressionModel():
    """
    A simple linear regression model implemented from scratch using numpy

    Args:
    - num_weights (int): The number of weights in the model
    - epochs (int): The number of training steps/epochs

    Attributes:
    - bias: (float): The bias of the model
    - weights: (numpy.ndarray): The weights of the model

    """
    def __init__(self, num_weights):
        """
       Initializes the weights and biases of the model

       Args:
        - num_weights (int): Defines the number of weights of the model

       Attributes:
        - bias (float): Bias term of the model
        - weights (numpy ndarray): Array containing the weights of the model

        """
        self.bias = 0.0
        self.weights = np.random.randn(num_weights)

    def predict(self, x):

        """
        Computes the prediction of the model

        Args:
        - x (numpy ndarray): Numpy array containing the input features

        Returns:
        - scalar (float): The predicted value

        """
        prediction = np.dot(x, self.weights) + self.bias
        print(f'Prediction: {prediction}')
        return prediction

    def compute_loss(self, x, y):
        yhat = self.pred(x)
        J = np.mean((yhat - y)**2)
        return J

    def train(self, x, y, learning_rate=0.01, epochs=100):
        for epoch in range(epochs):
            yhat = self.pred(x)
            loss = self.compute_loss(x, y)
            gradients = self.compute_gradients(x, yhat, y)
            self.training_step(gradients, learning_rate)
            print(f'[{epoch}] Loss: {loss}')
    def compute_gradients(self, x, yhat, y):
        dJ_dw = 2 * np.dot((yhat - y), x)
        dJ_db = 2 * (yhat - y)
        return dJ_dw, dJ_db

    def training_step(self, gradients, learning_rate):
        dJ_dw, dJ_db = gradients
        self.weights = self.weights - learning_rate * dJ_dw
        self.bias = self.bias - learning_rate * dJ_db

