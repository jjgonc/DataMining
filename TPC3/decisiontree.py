import numpy as np
from collections import Counter

class DecisionTree:
    def __init__(self, max_depth=None, max_features = None, min_samples_split=2, min_samples_leaf=1, criterion='gini', pre_prune='size', pre_pruning_threshold = 10, post_prune=None, post_pruning_threshold=0.01):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.pre_prune = pre_prune    # Pre-Prunning -> 'size'; 'independence' or 'max_depth'
        self.pre_pruning_threshold = pre_pruning_threshold
        self.post_prune = post_prune  # Post-Prunning -> 'pessimistic_error' or 'reduced_error'
        self.post_pruning_threshold = post_pruning_threshold

    def fit(self, X, y):
        self.tree = self._grow_tree(X, y)       # build the tree
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.tree) for x in X])     # predict the labels
    

    def _grow_tree(self, X, y, depth=0):
        
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))
        
        if len(X) == 0:
            return None
        
        # Check for homogeneity
        if len(np.unique(y)) == 1:
            return y[0]
        

        # Check for pre-pruning conditions
        if self.pre_prune == 'independence' and n_features == 1:
            return self._most_common_label(y)
        if self.pre_prune == 'max_depth' and depth >= self.max_depth:
            return self._most_common_label(y)
        if self.pre_prune ==  'size' and len(X) < self.pre_pruning_threshold:
            return self._most_common_label(y)
        
        # Stopping criteria
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split or n_samples < self.min_samples_leaf):
            leaf_value = self._most_common_label(y)
            return leaf_value
        
        # Grow tree
        feature_idx, threshold = self._best_criteria(X, y, depth)  # find the best split

        # Check for post-pruning conditions
        if self.post_prune == 'pessimistic_error' and self._is_pessimistic_error_prunable(X, y, best_attribute, best_threshold):
            return self._most_common_label(y)
        elif self.post_prune == 'reduced_error' and self._is_reduced_error_prunable(X, y, best_attribute, best_threshold):
            return self._most_common_label(y)

        # Split data based on best attribute and threshold
        left_idx, right_idx = self._split(X[:, feature_idx], threshold) # split the data
        left = self._grow_tree(X[left_idx, :], y[left_idx], depth + 1)  # grow the left subtree
        right = self._grow_tree(X[right_idx, :], y[right_idx], depth + 1)   # grow the right subtree

        return {'feature_idx': feature_idx, 'threshold': threshold, 'left': left, 'right': right}

    
    def _best_criteria(self, X, y, depth):
        nFeatures = X.shape[1]  # number of features
        best_impurity = np.inf
        split_idx = None
        split_threshold = None

        if self.max_features is not None:
            features_indices = np.random.choice(range(X.shape[1]), size=self.max_features, replace=False)
        else:
            features_indices = range(X.shape[1])

        for idx in features_indices:
            X_column = X[:, idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                _, y_right = self._split(X_column, threshold)

                ''' 
                # Pre-pruning
                if self.pre_prune == 'independence': # class distribution is independent of the avalailable features
                    p_left = len(X_column[X_column <= threshold])/len(X_column)
                    p_right = len(X_column[X_column > threshold])/len(X_column)
                    n_left = len(y_right)
                    n_right = len(y) - n_left
                    expected_left = len(y) * p_left
                    expected_right = len(y) * p_right
                    chi2 = ((n_left - expected_left) ** 2) / expected_left + ((n_right - expected_right) ** 2) / expected_right
                    if chi2 > 3.84:   # X^2 test
                       continue
                elif self.pre_prune == 'max_depth':
                    if depth >= self.max_depth:
                        continue
                elif self.pre_prune == 'size' and len(y_right) < self.pre_pruning_threshold:
                        continue
                
                '''
                
                if self.criterion == 'gini':
                    impurity = self.gini_index(y_right)
                elif self.criterion == 'entropy':
                    impurity = self.entropy(y_right)
                elif self.criterion == 'gain_ratio':
                    impurity_gain = self.gain_ratio(X_column, y_right)

                if (self.criterion == 'gain_ratio'):        # different case than entropy and giniIndex, where we need to maximize the gain ratio
                    if impurity_gain > best_impurity_gain:
                        best_impurity_gain = impurity_gain
                        split_idx = idx
                        split_threshold = threshold
                else:
                    if impurity < best_impurity:
                        best_impurity = impurity
                        split_idx = idx
                        split_threshold = threshold

        return split_idx, split_threshold

    def _split(self, X_column, split_threshold):
        left_idx = np.argwhere(X_column <= split_threshold).flatten()   # indices of the left subtree
        right_idx = np.argwhere(X_column > split_threshold).flatten()   # indices of the right subtree
        return left_idx, right_idx
    
    def _most_common_label(self, y):        # return the most common label
        counter = Counter(y)            # count the number of each label
        most_common = counter.most_common(1)[0][0]  
        return most_common
    
    def _traverse_tree(self, x, tree):  # traverse the tree to make a prediction
        if tree in [0, 1]:
            return tree
        
        feature_value = x[tree['feature_idx']]
        if feature_value <= tree['threshold']:
            return self._traverse_tree(x, tree['left'])
        return self._traverse_tree(x, tree['right'])
    
    ''' # Não estamos a usar de momento, mas teria como objetivo calcular o ganho do pruning
    def _get_pruning_gain(self, y, left_mask, right_mask, left_class, right_class):
        left_correct = np.sum(y[left_mask] == left_class)
        right_correct = np.sum(y[right_mask] == right_class)
        return (left_correct + right_correct) / len(y)
    ''' 
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

    # Post-Prunning ------- Checking for Pessimistic Error
    def _is_pessimistic_error_prunable(self, X, y, attribute, threshold):
        left_mask = X[:, attribute] < threshold
        right_mask = X[:, attribute] >= threshold
        
        p = len(left_mask) / len(y)
        error = self._get_pessimistic_error(y, p)
        return error < self.post_pruning_threshold

    # Post-Prunning ------- Checking for Reduced Error    
    def _is_reduced_error_prunable(self, X, y, attribute, threshold):
        left_mask = X[:, attribute] < threshold
        right_mask = X[:, attribute] >= threshold
        
        left_error = self._get_reduced_error(y[left_mask])
        right_error = self._get_reduced_error(y[right_mask])
        total_error = (len(y[left_mask]) * left_error + len(y[right_mask]) * right_error) / len(y)        
        return total_error < self.post_prune

    # Post-Prunning ------- Calculating Pessimistic Error
    def _get_pessimistic_error(self, y, p):
        return p + 1.96 * np.sqrt((p * (1 - p)) / len(y))
    
    # Post-Prunning ------- Calculating Reduced Error
    def _get_reduced_error(self, y):
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.max(counts) / np.sum(counts)
    

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
    
    clf = DecisionTree(max_depth=10, criterion='gain_ratio')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("Accuracy:", accuracy)
