import numpy as np
from typing import Callable
import f_classif
import sys
sys.path.append('../TPC1')
from tpc1 import Dataset

# class SelectKBest:
#     def __init__(self, score_func: Callable = f_classif, k: int=10):
#         if k <= 0:
#             raise ValueError("k must be positive")
        
#         self.k = k
#         self.p = None
#         self.F = None
#         self.score_func = score_func
        

#     def fit(self, dataset: Dataset) -> 'SelectKBest':
#         self.F, self.p = self.score_func(dataset)
#         return self

#     def transform(self, dataset: Dataset) -> Dataset:
#         indexes = np.argsort(self.F)[-self.k:]
#         features = np.array(dataset.getFeatureNames())[indexes]
#         return Dataset(x=dataset.getX()[:, indexes], y=dataset.getY(), feature_names=list(features), label_names=dataset.getLabelNames())

#     def fit_transform(self, dataset: Dataset) -> Dataset:
#         self.fit(dataset)
#         return self.transform(dataset)
    
# if __name__ == "__main__":
#     x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
#     y = np.array([0, 0, 0])

#     ds = Dataset(x, y, feature_names=['a', 'b', 'c'], label_names=['d'])

#     sb = SelectKBest(score_func=f_classif, k=2)
#     res = sb.fit_transform(ds)
#     print(res.getLabelNames())


import numpy as np

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
    
    
from sklearn.feature_selection import f_regression

# create a sample dataset
X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([1, 2, 3])

# create a SelectKBest object and fit_transform the data
selector = SelectKBest(score_func=f_regression, k=2)
X_new = selector.fit_transform(X, y)

# print the selected features
print(X_new)