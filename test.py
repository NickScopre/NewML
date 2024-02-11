import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import numpy as np
from datagen import DataGenerator

from LinearRegression.linear_regression import LinearRegressionModel

POINTS = 100
FEATURES = 2
FIGURE_ROWS = 1
FIGURE_COLS = 1
NOISE = 0.25

mpl.rcParams['figure.figsize'] = [20.0, 10.0]

DG = DataGenerator()
df = DG.gen_linear_data(points=POINTS, features=FEATURES, noise=NOISE)
X = df[0]
Y = df[0]

LRM = LinearRegressionModel()
LRM.fit(X.to_frame(), Y)
y_pred = LRM.predict(X.to_frame())

x_range = np.linspace(0, POINTS, num=POINTS)

fig, axs = plt.subplots(nrows=FIGURE_ROWS, ncols=FIGURE_COLS, sharex=True, sharey=True)
plt.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.05) 

# Plot each feature against the common x_range
for feature_name in df.columns:
    axs.scatter(x_range, df[feature_name], label=feature_name, s=25, alpha=0.5, marker='o', edgecolor='none')
#_nolegend_
axs.set_xlabel('Point')
axs.set_ylabel('Feature Value')
axs.set_title('Generated Data')

plt.plot(x_range, y_pred, color='red', marker='x', linestyle='-', label='Perfect Fit')
plt.legend()
plt.show()
