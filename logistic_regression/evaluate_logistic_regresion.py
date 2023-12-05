import pickle
from data_loader import X_test, y_test

def compute_accuracy(true_labels, predicted_labels):
    """
    Compute the accuracy of predictions.

    Args:
    - true_labels (numpy ndarray): Array containing the true labels.
    - predicted_labels (numpy ndarray): Array containing the predicted labels.

    Returns:
    - accuracy (float): The accuracy of the predictions.
    """
    # Check if the length of the true labels and predicted labels are the same
    if len(true_labels) != len(predicted_labels):
        raise ValueError("Length of true labels and predicted labels must be the same.")

    correct_predictions = (true_labels == predicted_labels).sum() # Calculate the number of correct predictions

    accuracy = correct_predictions / len(true_labels) # Calculate the accuracy

    return str(accuracy * 100) + '%'

with open('saved_trained_model/trained_iris_model.pkl', 'rb') as f:
    trained_iris_model = pickle.load(f)

pred = trained_iris_model.predict(X_test)

print(f'pred: {pred}')
print(y_test)
print(f'Accuracy: {compute_accuracy(pred, y_test)}')