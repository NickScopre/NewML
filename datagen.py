import pandas as pd
import numpy as np

class DataGenerator:
    def __init__(self):
        pass


    # This section needs work, maybe a rewrite, but the data doesn't have a linearly relationship between features.
    # Generate a dataset of linearly related features
    def gen_linear_data(self, points=1000, features=2, noise=0.5):
        self.points = points
        self.features = features
        self.noise = noise

        data = np.zeros(dtype=float, shape=(self.features, self.points))

        # Generate individual datavalues according to y = mx + b +/- n
        for f in range(self.features):
            slope = np.random.uniform(-10, 10)
            intercept = np.random.uniform(-5, 5)
            for p in range(self.points):
                # Normal b/c of Central Limit Theorem
                data[f][p] = slope * (p/self.points) + intercept + np.random.normal(-self.noise, self.noise)

        df = pd.DataFrame(data).transpose()
        return df