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
        """
        Initialize the logistic regression model

        Args:
         - num_features (int): The number of features of the input

        """
        self.weights = np.random.randn(num_features)
        self.bias = 0.0

    def forward_pass(self, x):
        """
        Forward pass of the model - computes the output probability

        Args:
         - x (numpy ndarray): Input features for a single data point

        Returns:
         - a (float): output: Output probability between 0 and 1

        """
        z = np.dot(x, self.weights) + self.bias
        a = 1/(1+np.exp(-z))
        return a

    def train(self, X, y, learning_rate=0.05, epochs=101):
        """
        Train the logistic regression model

        Args:
         - X (numpy ndarray): Matrix containing the features of the data points
         - y (numpy ndarray): Array containing the true labels of the data points
         - learning_rate (float): Learning rate for gradient descent (default: 0.05)
         - epochs (int): Number of training epochs (default: 101)

        Returns:
         - self (LogisticRegression): Trained model

        """

        for epoch in range(epochs):
            total_loss = 0
            for x_i, y_i in zip(X, y):
                a_i = self.forward_pass(x_i)
                loss = -(y_i * np.log(a_i) + (1 - y_i) * np.log(1 - a_i))
                total_loss += loss
                dj_dw = (a_i - y_i)*x_i
                dj_db = a_i - y_i
                self.weights = self.weights - learning_rate*dj_dw
                self.bias = self.bias - learning_rate*dj_db
            if epoch % 10 == 0:
                print(f'[{epoch}] Avg Loss: {total_loss/len(y):.4f}' + ' Accuracy: ' + str(np.sum(self.predict(X) == y)/len(y) * 100) + '%')
        return self

    def predict(self, X):
        """
        Make predictions using the trained model

        Args:
        - X: (numpy ndarray): Matrix containing the features of the data points

        Returns:
        - y_hat (numpy ndarray): Predicted labels (0 or 1) for each data point

        """
        y_hat = []
        for xi in X:
            a = self.forward_pass(xi)
            y_hat_i = np.where(a >= 0.5, 1, 0)
            y_hat.append(y_hat_i.item())
        y_hat = np.array(y_hat)
        return y_hat


