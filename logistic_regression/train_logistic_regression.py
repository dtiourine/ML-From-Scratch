from logistic_regression_model import LogisticRegression
from data_loader import X_train, y_train
import pickle
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    iris_model = LogisticRegression(num_features=4) #Instantiate model with 4 features
    trained_iris_model = iris_model.train(X_train, y_train) #Train the model on X, using true labels y
    loss_hist = trained_iris_model.loss_log
    epoch_list = np.arange(trained_iris_model.epochs)
    plt.plot(epoch_list, loss_hist)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    with open('saved_trained_model/trained_iris_model.pkl', 'wb') as f:
        pickle.dump(trained_iris_model, f)