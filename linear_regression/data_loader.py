from sklearn import datasets
from sklearn.model_selection import train_test_split

diabetes_data = datasets.load_diabetes()

X = diabetes_data.data
y = diabetes_data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# print(f'Length of X_train: {len(X_train)}')
# print(f'Length of y_train: {len(y_train)}')

# print(f'Length of y_train: {len(X_test)}')
# print(f'Length of y_train: {len(y_test)}')