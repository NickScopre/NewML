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
            # target is a series
            self.y = target.values
        elif(isinstance(target, np.ndarray)):
            # target is a ndarray
            self.y = target
        elif(isinstance(target, list)):
            # target is a python list
            self.y = np.array(target)
        else:
            # input is invalid
            raise TypeError("Target Values must be a Pandas Series, Numpy 1D Array, or Python List.")
        
        # Initialize Weights
        num_points, num_features = self.X.shape
        self.weights = np.zeros(num_features)
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
            # Transpose: X.T
            # Mat Mult: @
            # Inverse: np.linalg.inv()

            # Ensure full rank, otherwise matrix has linearly dependent columns and is singular
            """
            A Matrix is invertible iff it is full rank,
            such that it has a nonzero values along the diagonal.
            Given that the input matrix is MxN, X_T @ X will be NxN
            Upon performing SVD, the matrix of singular values (Sigma) must have a 
            nonzero value for every element on the diagonal
            """
            singular_values = np.linalg.svd(self.X, compute_uv=False)
            for i in range(num_features):
                if singular_values[i][i] == 0:
                    

            # Compute coefficients via least squares solution
            self.weights = np.linalg.inv(self.X.T @ self.X) @ self.X.T @ self.y
            return self.weights


    def predict(self, input, target):
        self.weights
        pass