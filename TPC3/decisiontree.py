import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1, criterion='gini', prune=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.prune = prune
        self.tree = None
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        num_samples, num_features = X.shape
        num_classes = len(set(y))
        # stop criteria
        if (self.max_depth is not None and depth >= self.max_depth) or \
           (num_samples < self.min_samples_split) or \
           (num_samples < 2 * self.min_samples_leaf):
            leaf_value = self._leaf_value(y)
            return leaf_value
        # find best split
        best_split = None
        best_split_criterion = 1e10
        if self.criterion == 'entropy':
            root_criterion = self._entropy(y)
        else:
            root_criterion = self._gini(y)
        for feature_idx in range(num_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                y_left = y[X_column < threshold]
                y_right = y[X_column >= threshold]
                if len(y_left) == 0 or len(y_right) == 0:
                    continue
                if self.criterion == 'entropy':
                    left_criterion = self._entropy(y_left)
                    right_criterion = self._entropy(y_right)
                else:
                    left_criterion = self._gini(y_left)
                    right_criterion = self._gini(y_right)
                weighted_criterion = (len(y_left) / num_samples) * left_criterion + (len(y_right) / num_samples) * right_criterion
                if weighted_criterion < best_split_criterion:
                    best_split = {'feature_idx': feature_idx, 'threshold': threshold, 
                                  'left_indices': X_column < threshold, 'right_indices': X_column >= threshold}
                    best_split_criterion = weighted_criterion
        # check if best split is not good enough to split
        if best_split_criterion > root_criterion:
            leaf_value = self._leaf_value(y)
            return leaf_value
        # split the tree and continue growing
        left_tree = self._grow_tree(X[best_split['left_indices']], y[best_split['left_indices']], depth + 1)
        right_tree = self._grow_tree(X[best_split['right_indices']], y[best_split['right_indices']], depth + 1)
        return {'feature_idx': best_split['feature_idx'], 'threshold': best_split['threshold'],
                'left': left_tree, 'right': right_tree}
    
    def _leaf_value(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _gini(self, y):
        counter = Counter(y)
        impurity = 1
        for label in counter:
            prob = counter[label] / len(y)
            impurity -= prob ** 2
        return impurity
'''
    def _entropy(self, y):
        counter = Counter(y)
        entropy = 

        Falta Gain Ratio
        Falta resollução de conflitos dos slides
        
    def main():

main()


'''