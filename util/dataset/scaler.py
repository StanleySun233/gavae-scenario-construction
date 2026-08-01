import numpy as np


EPS = 1e-18


class TSFeatureWiseScaler:
    def __init__(self, feature_range=(0, 1)):
        self._min_v, self._max_v = feature_range

    def fit(self, x):
        feature_dim = x.shape[2]
        self.mins = np.zeros(feature_dim)
        self.maxs = np.zeros(feature_dim)
        for i in range(feature_dim):
            self.mins[i] = np.min(x[:, :, i])
            self.maxs[i] = np.max(x[:, :, i])
        return self

    def transform(self, x):
        return ((x - self.mins) / (self.maxs - self.mins + EPS)) * (self._max_v - self._min_v) + self._min_v

    def inverse_transform(self, x):
        y = np.array(x, copy=True)
        y -= self._min_v
        y /= self._max_v - self._min_v
        y *= self.maxs - self.mins + EPS
        y += self.mins
        return y

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
