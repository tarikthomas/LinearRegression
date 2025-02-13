import numpy as np
from sklearn.metrics import mean_squared_error
class MyLinearRegression:
    def __init__(self, learningRate = 0.001, n_iterations = 1000):
        self.learningRate = learningRate # the rate at which the weights are updated
        self.n_iterations = n_iterations # the number of iterations the model will run
        
        # the weights and the model reperesnent the line in the form y = wx + b
        self.weights = None 
        self.bias = None

    def fit(self, X, y):
        # samples are the number of rows, features are the number of 
        # columns and also the specific features we are trying to use in the model

        n_samples, n_features = X.shape 
        self.weights = np.zeros(n_features) # initialize the weights to zero making sure the number of weights is equal to the number of features
        self.bias = 0 # initialize the bias to zero same for all


        for _ in range(self.n_iterations):

            # gradient descent
            y_pred = np.dot(X, self.weights) + self.bias # the y = wx + b equation

            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) # the derivative of the weights
            db = (1/n_samples) * np.sum(y_pred - y) # the derivative of the bias

            # update the weights and bias
            self.weights = self.weights - self.learningRate * dw
            self.bias = self.bias - self.learningRate * db

    def prediction(self, X):
        return np.dot(X, self.weights) + self.bias
    



    def rmse(self, X, y):

        predictions = self.prediction(X)

        return np.sqrt(np.mean((predictions - y) ** 2))

