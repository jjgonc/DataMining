import numpy as np
from typing import Callable
from sklearn.feature_selection import f_regression
import sys
sys.path.append('../TPC1')
from tpc1 import Dataset

class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.pvalues = None
        self.scores = None
        self.featuresSelected = None
    
    def fit(self, X, y):
        self.scores, self.p_values = self.score_func(X, y)
        
    def transform(self, X):
        mask = np.argsort(self.p_values)[:self.k]
        self.featuresSelected = mask
        return X[:, mask]
    
    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)
    
    


# create a sample dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, 3])

kBest = SelectKBest(score_func=f_regression, k=2)
res = kBest.fit_transform(X, y)

print(res)