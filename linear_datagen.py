import pandas as pd
import numpy as np

class LinearDataGenerator:
    def __init__(self, points=1000, features=2, noise=0.5):
        self.points = points
        self.features = features
        self.noise = noise

    # Generate a dataset of linearly related features
    def gen_linear_data(self):
        data = np.zeros(dtype=float, shape=(self.features, self.points))

        # Generate individual datavalues according to y = mx + b +/- n
        for f in range(self.features):
            slope = np.random.uniform(-1, 1)
            intercept = np.random.uniform(-1, 1)
            for p in range(self.points):
                # Normal b/c of Central Limit Theorem
                data[f][p] = slope * (p/self.points) + intercept + np.random.normal(-self.noise, self.noise)

        df = pd.DataFrame(data).transpose()
        return df
