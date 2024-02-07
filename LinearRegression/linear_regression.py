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
            self.X = input.values
        elif(isinstance(input, np.ndarray)):
            # input is a ndarray
            self.X = input
        elif(isinstance(input, list)):
            # input is a python list
            # CHECK IF CONTENTS OF LIST ARE LISTS OF VALID VALUES
            self.X = np.array(input)
        else:
            # input is invalid
            raise TypeError("Input Features must be a Pandas Dataframe, Numpy 2D Array, or Python List.")

        # Target
        if(isinstance(target, pd.Series)):
            # input is a series
            self.X = target.values
        elif(isinstance(target, np.ndarray)):
            # input is a ndarray
            self.X = target
        elif(isinstance(target, list)):
            # input is a python list
            self.X = np.array(target)
        else:
            # input is invalid
            raise TypeError("Target Values must be a Pandas Series, Numpy 1D Array, or Python List.")
        
        # Initialize Weights
        num_points, num_features = self.X.shape
        self.weights = np.zeros(num_features)
        self.bias = 0

        if(self.iterative):
            # Use batch gradient descent, learning rate, and number of iterations


            # Gradient Descent
                # Compute Predicted Values
                # Check for Convergeance
                # Calculate Gradient of Cost
                # Update Weights

            return self.weights

        else:
            # Compute Optimal coeffiecients regardless of number of iterations
            pass


    def predict(self, input, target):
        self.weights
        pass