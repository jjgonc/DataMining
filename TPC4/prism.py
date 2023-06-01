import numpy as np
from collections import Counter

class PRISMClassifier:
    def __init__(self, max_rules=10):
        self.max_rules = max_rules
        self.rules = []

    def fit(self, X_train, y_train):
        """
        Fit the PRISM classifier to the training data.

        Parameters:
            X_train (ndarray): Training features.
            y_train (ndarray): Training labels.

        Returns:
            None
        """
        while X_train.shape[0] > 0 and len(self.rules) < self.max_rules:
            best_rule = self.get_best_rule(X_train, y_train)
            if best_rule is None:
                break
            self.rules.append(best_rule)

            X_train, y_train = self.remove_patterns(X_train, y_train, best_rule)

    def get_best_rule(self, X_train, y_train):
        """
        Find the best rule for the PRISM classifier based on the training data.

        Parameters:
            X_train (ndarray): Training features.
            y_train (ndarray): Training labels.

        Returns:
            best_rule (tuple): The best rule found.
        """
        best_rule = None
        max_accuracy = -10

        for feature_index in range(X_train.shape[1]):
            feature_values = np.unique(X_train[:, feature_index])
            for feature_value in feature_values:
                for class_label in np.unique(y_train):
                    rule = (feature_index, feature_value, class_label)
                    accuracy = self.evaluate_pattern(X_train, y_train, rule)
                    if accuracy > max_accuracy:
                        best_rule = rule
                        max_accuracy = accuracy

        return best_rule

    def evaluate_pattern(self, X, y, pattern):
        """
        Evaluate the accuracy of a rule/pattern on the given data.

        Parameters:
            X (ndarray): Input features.
            y (ndarray): Input labels.
            pattern (tuple): The rule/pattern to evaluate.

        Returns:
            accuracy (float): The accuracy of the rule/pattern.
        """
        pattern_index, pattern_value, pattern_class = pattern
        matches_pattern = X[:, pattern_index] == pattern_value
        return np.mean(y[matches_pattern] == pattern_class)

    def remove_patterns(self, X, y, pattern):
        """
        Remove the instances covered by a rule/pattern from the data.

        Parameters:
            X (ndarray): Input features.
            y (ndarray): Input labels.
            pattern (tuple): The rule/pattern to remove.

        Returns:
            X_updated (ndarray): Updated features after removing covered instances.
            y_updated (ndarray): Updated labels after removing covered instances.
        """
        pattern_index, pattern_value, _ = pattern
        uncovered = X[:, pattern_index] != pattern_value
        return X[uncovered], y[uncovered]

    def predict(self, X):
        """
        Predict the labels for the input data.

        Parameters:
            X (ndarray): Input features.

        Returns:
            predictions (list): Predicted labels for the input data.
        """
        predictions = []
        for instance in X:
            instance_predictions = []
            for rule in self.rules:
                feature_index, feature_value, class_label = rule
                if instance[feature_index] == feature_value:
                    instance_predictions.append(class_label)
            if instance_predictions:
                class_counts = Counter(instance_predictions)
                most_common_class = class_counts.most_common(1)[0][0]
                predictions.append(most_common_class)
            else:
                predictions.append(None)
        return predictions

    def accuracy_score(self, y_true, y_pred):
        """
        Compute the accuracy score.

        Parameters:
            y_true (ndarray): True labels.
            y_pred (ndarray): Predicted labels.

        Returns:
            accuracy (float): Accuracy score.
        """
        return np.mean(y_true == y_pred)

    def __repr__(self):
        """
        Generate a string representation of the PRISM classifier rules.

        Returns:
            repr_str (str): String representation of the classifier rules.
        """
        repr_str = "PRISM Classifier Rules:\n"
        for i, rule in enumerate(self.rules):
            repr_str += f"Rule {i+1}: {rule}\n"
        return repr_str


def test():
    # Training data
    X_train = np.array([
        ["Overcast", "Hot", "Normal", "Weak"],
        ["Overcast", "Mild", "High", "Strong"],
        ["Sunny", "Mild", "Normal", "Strong"],
        ["Rain", "Mild", "Normal", "Weak"],
        ["Sunny", "Cool", "Normal", "Weak"],
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Sunny", "Mild", "High", "Weak"],
        ["Rain", "Cool", "Normal", "Strong"],
        ["Rain", "Cool", "Normal", "Weak"],
        ["Rain", "Mild","High","Weak"]
        #[1, 1, 1, 1],
        #[0, 0, 0, 0]
    ])
    y_train = np.array([0, 0, 0, 0, 0, 0,1,1, 0, 0])

    # Test data
    X_test = np.array([
        ["Overcast", "Cool", "Normal", "Strong"],
        ["Rain", "Cool", "Normal", "Weak"],
         ["Sunny", "Mild", "High", "Weak"],
        ["Rain", "Cool", "Normal", "Strong"],
        ["Sunny", "Hot","High","Strong"]
    ])
    y_test = np.array([0, 0, 1, 1,1])

    prism = PRISMClassifier(max_rules=10)  # Set the maximum number of rules to 10
    prism.fit(X_train, y_train)
    print(prism)

    predictions = prism.predict(X_test)
    print("Predictions:", predictions)
    print("True labels:", y_test)

    accuracy = prism.accuracy_score(y_test, predictions)
    print("Accuracy:", accuracy)


if __name__ == "__main__":
    test()
