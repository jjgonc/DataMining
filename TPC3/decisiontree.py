import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, max_features = None, min_samples_split=2, min_samples_leaf=1, criterion='gini', pre_prune=None, post_prune=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.pre_prune = pre_prune
        self.post_prune = post_prune
    
    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)       # build the tree
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])     # predict the labels
    

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        # Grow tree
        feature_idx, threshold = self._best_criteria(X, y)  # find the best split
        left_idx, right_idx = self._split(X[:, feature_idx], threshold) # split the data
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)  # grow the left subtree
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)   # grow the right subtree

        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left, 'right': right}


    def _best_criteria(self, X, y):
        nFeatures = X.shape[1]  # number of features
        best_impurity = np.inf
        split_bestFeature, split_bestThreshold = None, None

        if self.max_features is not None:
            features_indices = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
        else:
            features_indices = range(X.shape[1])

        for idx in features_indices:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                _, y_right = self._split(X_column, threshold)
                if self.criterion == 'gini':
                    impurity = self.gini_index(y_right)
                elif self.criterion == 'entropy':
                    impurity = self.entropy(y_right)
                elif self.criterion == 'gain_ratio':
                    impurity = self.gain_ratio(X_column, y_right)
                
                if impurity < best_impurity:
                    best_impurity = impurity
                    split_idx = idx
                    split_threshold = threshold

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
        entropy = -np.sum(probabilities * np.log2(probabilities))   # -sum(p(x)log2(p(x)))
        return entropy
    
    def gini_index(self, label):
        counts = np.unique(label, return_counts=True)[1]
        proportions = counts / len(label)
        gini = 1 - np.sum(proportions ** 2)                   # 1 - sum(p(x)^2)
        return gini

    def gain_ratio(self, feature, labels):
        n = len(labels)
        values, counts = np.unique(feature, return_counts=True)
        entropy = self.entropy(labels)
        IV = - np.sum((counts / n) * np.log2(counts / n))             # Intrinsic value = -sum(p(x)log2(p(x)))
        IG = entropy                                                  # Information gain 
        for value, count in zip(values, counts):                
            subset_labels = labels[feature == value]
            IG -= (count / n) * self.entropy(subset_labels)
        return IG / IV if IV != 0 else 0                              # Gain ratio = Information gain / Intrinsic value 



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