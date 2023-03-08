from scipy.stats import f_oneway  
import numpy as np
from TPC1.tpc1 import Dataset

class F_Classif:
    def __init__(self, pvalues, fvalues):
        self.pvalues = None
        self.fvalues = None

    def fit(self, dataset: Dataset, y: np.ndarray) -> 'F_Classif':
        groups = np.unique(y)
        X_groups = [dataset.X[y == group] for group in groups]
        pv, fv = []
        for i in range(dataset.X.shape[1]):
            fv.append(f_oneway(*[X[:, i] for X in X_groups])[0])
            pv.append(f_oneway(*[X[:, i] for X in X_groups])[1])
        self.pvalues = np.array(pv)
        self.fvalues = np.array(fv)
        return self
    
    def transform(self, dataset: Dataset, threshold: float) -> Dataset:
        X = dataset.X
        features_mask = self.pvalues < threshold
        X = X[:, features_mask]
        features = np.array(dataset.features)[features_mask]
        return Dataset(X=X, y=dataset.y, features=list(features), label=dataset.label)
    
    def fit_transform(self, dataset: Dataset, y: np.ndarray, threshold: float) -> Dataset:
        self.fit(dataset, y)
        return self.transform(dataset, threshold)