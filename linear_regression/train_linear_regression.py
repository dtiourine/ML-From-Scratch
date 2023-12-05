from linear_regression_model import LinearRegression
import pickle
from data_loader import X_train, y_train
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    diabetes_model = LinearRegression(num_features=10) #Instantiate model with 4 features
    trained_diabetes_model = diabetes_model.train(X_train, y_train, learning_rate=0.05, epochs=150) #Train the model on X, using true labels y
    loss_hist = trained_diabetes_model.loss_log
    epoch_list = np.arange(trained_diabetes_model.epochs)
    plt.plot(epoch_list, loss_hist)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    with open('saved_trained_model/trained_diabetes_model.pkl', 'wb') as f:
        pickle.dump(trained_diabetes_model, f)
