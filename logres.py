# Import necessary modules
import time
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

def run_augmented_data(name):
    """
    Loads the augmented data file of the given name, separating it into
    training and testing data sets. Afterwards, runs the methods for
    finding the best model and plotting the coefficients.
    
    :param name: the name of the data file we're training the model on
    :type name: string
    :return: None
    """

    # Load the data from the npz files
    data = np.load(name)

    X_train = data['X_train']
    y_train = data['y_train']

    X_test = data['X_val']
    y_test = data['y_val']

    # Print the shape of the training and validation sets
    print(X_train.shape)
    print(X_test.shape)

    # Print the shape of the training and validation sets
    print(y_train.shape)
    print(y_test.shape)

    model = find_best(X_train, X_test, y_train, y_test)
    plot(model)

def find_best(X_train, X_test, y_train, y_test):
    """
    Finds the best logistic regression model. Prints its accuracy and the
    time it took to find it.
    
    :param X_train: The training data
    :type X_train: ndarray
    :param X_test: The testing data
    :type X_test: ndarray
    :param y_train: The training labels
    :type y_train: ndarray
    :param y_test: The testing labels
    :type y_test: ndarray
    :return: The best model
    :rtype: object
    """

    # Create a logistic regression model and find the best parameters
    startTime = time.time()
    parameters = {'C':[0.001, .01, .1, 1, 10, 20, 50]}
    model = LogisticRegression(max_iter=50000)
    cv = GridSearchCV(model, parameters, verbose=3)
    cv.fit(X_train, y_train)
    cv.best_params_

    # Print the model's accuracy
    best_model = cv.best_estimator_
    print("Accuracy: ", best_model.score(X_test, y_test))
    print("Time to train logistic regression: " + str(time.time() - startTime) + " seconds")

    return best_model

def plot(best_model):
    """
    Plots the index and magnitude of each coefficient in the given
    model

    :param best_model: The model to plot
    :type best_model: object
    :return: None
    """

    # Plot the model's coefficients
    plt.plot(best_model.coef_.T, 'o')

    # Add labels and show the plot
    plt.xlabel('Coefficient index')
    plt.ylabel('Coefficient magnitude')
    plt.show()