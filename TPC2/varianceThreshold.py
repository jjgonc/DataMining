import numpy as np
# from TPC1.tpc1 import Dataset
import sys
sys.path.append('../TPC1')
from tpc1 import Dataset

class VarianceThreshold:
    def __init__(self, threshold):
        if threshold < 0:
            raise ValueError("Threshold must be 0 or higher")
        
        self.threshold = threshold
        self.variances = None

    # Fit the VarianceThreshold model according to the given training data.
    def fit(self, dataset: Dataset) -> 'VarianceThreshold':
        self.variance = np.var(dataset.getX(), axis=0)   # calculate the variance of each feature along the columns
        return self

    # It removes all features whose variance is not high enough to meet the threshold.
    def transform(self, dataset: Dataset) -> Dataset:
        X = dataset.getX()

        features_mask = self.variance > self.threshold  # create a mask of features to keep
        X = X[:, features_mask]                    # apply mask to X
        features = np.array(dataset.getFeatureNames())[features_mask]    # update the features list, leaving only the ones that passed through filtering
        return Dataset(x=X, y=dataset.y, feature_names=list(features), label_names=dataset.getLabelNames())

    # Fit to data, then transform it.
    def fit_transform(self, dataset: Dataset) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset)
    

if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([0, 0, 0])

    ds = Dataset(x, y, feature_names=['a', 'b', 'c'], label_names=['d'])
    vt = VarianceThreshold(threshold=0.5)
    res = vt.fit_transform(ds)
