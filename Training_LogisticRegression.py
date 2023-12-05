import numpy as np
from LogisticRegression import LogisticRegression
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
y = iris.target
X = X[y !=2] #Turn into binary classification task by only looking at the first two classes
y = y[y !=2] #Turn into binary classification task by only looking at the first two classes

#print(X)
#print(y)

iris_model = LogisticRegression(num_features=4)
iris_model.train(X, y)
yhat = iris_model.predict(X)

def accuracy(yhat, y):
    acc = np.sum(yhat == y)/len(y) * 100
    return 'Accuracy: '+ str(acc) + '%'

print(accuracy(yhat, y))