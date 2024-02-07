import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from ScopreML.datagen import DataGenerator

from LinearRegression.linear_regression import LinearRegressionModel

POINTS = 5000
FEATURES = 3
NOISE = 2

DG = DataGenerator(points=POINTS, features=FEATURES, noise=0.5)
df = DG.gen_linear_data()

X = df[0]
X2 = df[1]
Y = df[2]

LRM = LinearRegressionModel()
LRM.fit(X, Y)
LRM.predict(X2, Y)

