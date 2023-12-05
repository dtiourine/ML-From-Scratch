import pickle
from data_loader import X_test, y_test

def mean_squared_error(actual, predicted):
    """
    Calculate the Mean Squared Error (MSE) between actual and predicted values.

    Parameters:
    actual (list or numpy array): The actual target values.
    predicted (list or numpy array): The predicted target values.

    Returns:
    float: The Mean Squared Error between actual and predicted values.
    """
    if len(actual) != len(predicted):
        raise ValueError("Input lists must have the same length")

    squared_errors = [(actual[i] - predicted[i]) ** 2 for i in range(len(actual))]
    mse = sum(squared_errors) / len(actual)
    return mse

with open('saved_trained_model/trained_diabetes_model.pkl', 'rb') as f:
    trained_diabetes_model = pickle.load(f)

pred = trained_diabetes_model.predict(X_test)

MSE = mean_squared_error(y_test, pred)
print(f'MSE: {MSE}')