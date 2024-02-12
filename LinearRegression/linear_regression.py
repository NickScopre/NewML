import pandas as pd
import numpy as np

class LinearRegressionModel:
    def __init__(self, iterative=False, eta = 0.01, iterations=100):
        self.iterative = iterative
        self.eta = eta
        self.iterations = iterations

    def fit(self, input, target):
        # Handle Python Lists, Numpy Ndarrays, and Pandas Dataframes
        # Input
        if(isinstance(input, pd.DataFrame)):
            # input is a df
            self.X = input.to_numpy()
        elif(isinstance(input, np.ndarray)):
            # input is a ndarray
            self.X = input
        elif(isinstance(input, list)):
            # input is a python list
            # CHECK IF CONTENTS OF LIST ARE LISTS OF VALID VALUES
            self.X = np.array(input)
        else:
            # input is invalid
            raise TypeError("Error 1: Input Features must be a Pandas Dataframe, Numpy 2D Array, or Python List.")

        # Target
        if(isinstance(target, pd.Series)):
            # target is a series
            self.y = target.to_numpy()
        elif(isinstance(target, np.ndarray)):
            # target is a ndarray
            self.y = target
        elif(isinstance(target, list)):
            # target is a python list
            self.y = np.array(target)
        else:
            # input is invalid
            raise TypeError("Error 2: Target Values must be a Pandas Series, Numpy 1D Array, or Python List.")
        
        # Initialize Weights
        num_points, num_features = self.X.shape
        self.weights = np.zeros(num_features, dtype=float)
        self.bias = 0

        if(self.iterative):
            # Use batch gradient descent, learning rate, and number of iterations


            ## Gradient Descent ##
            # Compute Predicted Values
            # Check for Convergeance
            # Calculate Gradient of Cost
            # Update Weights

            return self.weights

        else:
            ## Compute Optimal Coefficients ##

            # Compute coefficients via least squares solution
            self.weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
            return self.weights


    def predict(self, input):
        y_pred = np.dot(input, self.weights)
        return y_pred