import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from DataGen.linear_datagen import LinearDataGenerator

POINTS = 500
FEATURES = 5

LDG = LinearDataGenerator(points=POINTS, features=FEATURES, noise=0.5)
df = LDG.gen_linear_data()

x_range = np.linspace(df.min().min(), df.max().max(), num=POINTS)

fig, ax = plt.subplots()


# Plot each feature against the common x_range
for feature_name in df.columns:
    ax.scatter(x_range, df[feature_name], label="Feature"+str(feature_name), s=25, alpha=0.5)

plt.xlabel('Common X-axis')
plt.ylabel('Feature Values')
plt.title('Scatter Plot of Features')
plt.legend()
plt.show()
