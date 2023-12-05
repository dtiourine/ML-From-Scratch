import numpy as np
from LogisticRegression import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris() #Load in the Iris dataset from scikit learn

# Retrieve the feature matrix (X) and target labels (y)
X = iris.data
y = iris.target

# Create a binary classification task by selecting only the first two classes
X = X[y !=2]
y = y[y !=2]

#print(X)
#print(y)

iris_model = LogisticRegression(num_features=4) #Instantiate model with 4 features

iris_model.train(X, y) #Train the model on X, using true labels y
