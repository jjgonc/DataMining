import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from collections import Counter

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    BROWN = "\033[0;33m"
    DARK_GRAY = "\033[1;30m"

class Node:
    def __init__(self, split_threshold=None, split_feature=None, label=None, impurity=None):
        self.split_threshold = split_threshold
        self.split_feature = split_feature
        self.label = label
        self.impurity = impurity
        self.left = None
        self.right = None

class DecisionTreeClassifier:
    def __init__(self, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, conflict_resolution='majority'):
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.conflict_resolution = conflict_resolution
        self.tree = None

    def _calculate_impurity(self, y):
        if self.criterion == 'gini':
            class_counts = np.bincount(y)
            class_probabilities = class_counts / len(y)
            impurity = 1 - np.sum(class_probabilities**2)
            return impurity
        elif self.criterion == 'entropy':
            class_counts = np.bincount(y)
            class_probabilities = class_counts / len(y)
            impurity = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
            return impurity
        elif self.criterion == 'gain_ratio':
            class_counts = np.bincount(y)
            class_probabilities = class_counts / len(y)
            impurity = -np.sum(class_probabilities * np.log2(class_probabilities + 1e-10))
            return impurity
        else:
            raise ValueError("Invalid criterion. Supported criteria are 'gini', 'entropy', and 'gain_ratio'.")

    def _select_best_split(self, X, y):
        best_impurity = float('inf')
        best_split_feature = None
        best_split_threshold = None

        for feature in range(X.shape[1]):
            unique_values = np.unique(X[:, feature])
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2

            for threshold in thresholds:
                y_left = y[X[:, feature] <= threshold]
                y_right = y[X[:, feature] > threshold]

                impurity = (len(y_left) * self._calculate_impurity(y_left) +
                            len(y_right) * self._calculate_impurity(y_right)) / len(y)

                if impurity < best_impurity:
                    best_impurity = impurity
                    best_split_feature = feature
                    best_split_threshold = threshold

        return best_split_feature, best_split_threshold, best_impurity

    def _preprune_by_size(self, X, y):                      # Apply pre-pruning by size
        if len(X) < self.min_samples_split:                 # If the number of samples is less than the minimum number of samples to split
            if self.conflict_resolution == 'majority':      # If the conflict resolution is majority
                label = Counter(y).most_common(1)[0][0]     # Get the most common label
                return Node(label=label)                    # Return a node with the most common label

        return None

    def _preprune_by_depth(self, X, y, depth):
        if self.max_depth is not None and depth >= self.max_depth:  # If the maximum depth is not None and the current depth is greater than or equal to the maximum depth
            return Node(label=Counter(y).most_common(1)[0][0])      # Return a node with the most common label
        return None

    def _build_tree(self, X, y, depth):
        preprune_node = self._preprune_by_size(X, y)
        if preprune_node is not None:
            return preprune_node

        preprune_node = self._preprune_by_depth(X, y, depth)
        if preprune_node is not None:
            return preprune_node

        best_split_feature, best_split_threshold, best_impurity = self._select_best_split(X, y)

        node = Node(split_threshold=best_split_threshold, split_feature=best_split_feature, impurity=best_impurity)

        X_left = X[X[:, best_split_feature] <= best_split_threshold]
        y_left = y[X[:, best_split_feature] <= best_split_threshold]
        if len(X_left) >= self.min_samples_leaf:
            node.left = self._build_tree(X_left, y_left, depth + 1)

        X_right = X[X[:, best_split_feature] > best_split_threshold]
        y_right = y[X[:, best_split_feature] > best_split_threshold]
        if len(X_right) >= self.min_samples_leaf:
            node.right = self._build_tree(X_right, y_right, depth + 1)

        return node

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _predict_instance(self, x, node):
        if node.label is not None:
            return node.label

        if x[node.split_feature] <= node.split_threshold:
            return self._predict_instance(x, node.left)
        else:
            return self._predict_instance(x, node.right)

    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self._predict_instance(x, self.tree)
            predictions.append(prediction)
        return np.array(predictions)

    def __repr__(self):
        return self._print_tree(self.tree)

    def _print_tree(self, node, depth=0):
        if node.label is not None:
            return "  " * depth + bcolors.OKGREEN + "Leaf: Class " + str(node.label) + bcolors.ENDC

        representation = "  " * depth + bcolors.DARK_GRAY + "Node(split_threshold={}, split_feature={}, impurity={})\n".format(
            node.split_threshold, node.split_feature, node.impurity) + bcolors.ENDC

        if node.left is not None:
            representation += self._print_tree(node.left, depth + 1) + "\n"

        if node.right is not None:
            representation += self._print_tree(node.right, depth + 1)

        return representation

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    criterionType = 'entropy'
    # criterionType = 'gini'
    # criterionType = 'gain_ratio'

    dt = DecisionTreeClassifier(criterion=criterionType, max_depth=3, min_samples_split=2, min_samples_leaf=1)
    dt.fit(X_train, y_train)
    print(dt)

    predictions = dt.predict(X_test)
    print("Using criterion:", criterionType)
    # print("Predictions:", predictions)
    # print("Actual:", y_test)

if __name__ == '__main__':
    main()
