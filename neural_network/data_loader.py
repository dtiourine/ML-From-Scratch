import pandas as pd
import numpy as np

train = pd.read_csv('data/train.csv')
train = np.array(train)
np.random.shuffle(train)
m, n = train.shape

X_train = train[1:n]
y_train = train[0]