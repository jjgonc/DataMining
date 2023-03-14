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
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        # Grow tree
        feature_idx, threshold = self._best_criteria(X, y)
        left_idx, right_idx = self._split(X[:, feature_idx], threshold)
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)
        
        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left, 'right': right}
    
    def _best_criteria(self, X, y):
        m = X.shape[1]
        best_gini = 100
        split_idx, split_threshold = None, None

        if self.criterion == 'gini':
            for idx in range(m):
                X_column = X[:, idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    _, y_right = self._split(X_column, threshold)
                    gini = self.gini_index(y_right)
                    
                    if gini < best_gini:
                        best_gini = gini
                        split_idx = idx
                        split_threshold = threshold

        elif self.criterion == 'entropy':
            for idx in range(m):
                X_column = X[:, idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    _, y_right = self._split(X_column, threshold)
                    entropy = self.entropy(y_right)
                    
                    if entropy < best_gini:
                        best_gini = entropy
                        split_idx = idx
                        split_threshold = threshold

        elif self.criterion == 'gain_ratio':
            for idx in range(m):
                X_column = X[:, idx]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    _, y_right = self._split(X_column, threshold)
                    gain_ratio = self.gain_ratio(X_column, y_right)
                    
                    if gain_ratio > best_gini:
                        best_gini = gain_ratio
                        split_idx = idx
                        split_threshold = threshold
        
        # for idx in range(m):
        #     X_column = X[:, idx]
        #     thresholds = np.unique(X_column)
        #     for threshold in thresholds:
        #         _, y_right = self._split(X_column, threshold)
        #         gini = self.gini_index(y_right)
                
        #         if gini < best_gini:
        #             best_gini = gini
        #             split_idx = idx
        #             split_threshold = threshold
        
        return split_idx, split_threshold
    
    def _split(self, X_column, split_threshold):
        left_idx = np.argwhere(X_column <= split_threshold).flatten()
        right_idx = np.argwhere(X_column > split_threshold).flatten()
        return left_idx, right_idx
    
    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common
    
    def _traverse_tree(self, x, tree):
        if tree in [0, 1]:
            return tree
        
        feature_value = x[tree['feature_idx']]
        if feature_value <= tree['threshold']:
            return self._traverse_tree(x, tree['left'])
        return self._traverse_tree(x, tree['right'])
    
    def entropy(self, label):
        _, counts = np.unique(label, return_counts=True)
        probabilities = counts / len(label)
        entropy = -np.sum(probabilities * np.log2(probabilities))
        return entropy
    
    def gini_index(self, label):
        counts = np.unique(label, return_counts=True)[1]
        proportions = counts / len(label)
        gini = 1 - np.sum(proportions ** 2)
        return gini

    def gain_ratio(self, feature, labels):
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        H = self.entropy(labels)
        IV = - np.sum((counts / n) * np.log2(counts / n))
        IG = H
        for value, count in zip(values, counts):
            subset_labels = labels[feature == value]
            IG -= (count / n) * self.entropy(subset_labels)
        return IG / IV if IV != 0 else 0



if __name__ == '__main__':
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy
    
    data = datasets.load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)
    
    clf = DecisionTree(max_depth=10, criterion='entropy')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)