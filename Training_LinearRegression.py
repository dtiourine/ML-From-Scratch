import numpy as np
from LinearRegression import LinearRegression
from sklearn import datasets

house_prices = datasets.fetch_california_housing()

X = house_prices.data
y = house_prices.target

houseprice_model = LinearRegression(num_features=8)
print(houseprice_model.predict(X))
houseprice_model.train(X, y, learning_rate=0.001)