from logistic_regression_model import LogisticRegression
from data_loader import X_train, y_train
import pickle

if __name__ == "__main__":
    iris_model = LogisticRegression(num_features=4) #Instantiate model with 4 features
    trained_iris_model = iris_model.train(X_train, y_train) #Train the model on X, using true labels y

    with open('saved_trained_model/trained_iris_model.pkl', 'wb') as f:
        pickle.dump(trained_iris_model, f)