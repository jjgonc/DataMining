import numpy as np
from TPC1.tpc1 import Dataset

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        if threshold < 0:
            raise ValueError("Threshold must be 0 or higher")
        
        self.threshold = threshold
        self.variances = None

    # Fit the VarianceThreshold model according to the given training data.
    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variance = np.var(dataset.X, axis=0)   # calculate the variance of each feature along the columns
        return self

    # It removes all features whose variance is not high enough to meet the threshold.
    def transform(self, dataset: Dataset) -> Dataset:
        X = dataset.X

        features_mask = self.variance > self.threshold  # create a mask of features to keep
        X = X[:, features_mask]                    # apply mask to X
        features = np.array(dataset.features)[features_mask]    # update the features list, leaving only the ones that passed through filtering
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)

    # Fit to data, then transform it.
    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)