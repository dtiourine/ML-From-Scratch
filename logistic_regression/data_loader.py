from sklearn import datasets
from sklearn.model_selection import train_test_split

iris = datasets.load_iris() #Load in the Iris dataset from scikit learn

# Retrieve the feature matrix (X) and target labels (y)
X = iris.data
y = iris.target

# Create a binary classification task by selecting only the first two classes
X = X[y !=2]
y = y[y !=2]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# print(f'Length of X_train: {len(X_train)}')
# print(f'Length of y_train: {len(y_train)}')

# print(f'Length of y_train: {len(X_test)}')
# print(f'Length of y_train: {len(y_test)}')

# print(y_train)
