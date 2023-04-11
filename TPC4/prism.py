import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Prism:
    def __init__(self, min_support=0.1, max_rules=10):
        self.min_support = min_support
        self.max_rules = max_rules
        self.rules = []

    def fit(self, X, y):
        self.X = X
        self.y = y
        df = pd.concat([X, y], axis=1)
        for column in X.columns:
            df[column] = pd.qcut(df[column], q=10, duplicates='drop')

        self.rules = self._prism(df)

    def predict(self, X):
        y_pred = []
        default_class = pd.Series(self.y).mode().values[0]
        for _, row in X.iterrows():
            for rule in self.rules:
                if all([row[k] in v for k, v in rule['conditions'].items()]):
                    y_pred.append(rule['class'])
                    break
            else:
                y_pred.append(default_class)
        return y_pred

    def _prism(self, df):
        rules = []
        while len(rules) < self.max_rules:
            candidates = []
            for column in df.columns[:-1]:
                for value in df[column].unique():
                    df_subset = df[df[column] == value]
                    support = df_subset.iloc[:, -1].mean()
                    if support >= self.min_support:
                        conditions = {column: [value]}
                        class_ = df_subset.iloc[:, -1].mode().values[0]
                        candidates.append({'conditions': conditions, 'class': class_, 'support': support})

            if not candidates:
                break

            candidates.sort(key=lambda x: x['support'], reverse=True)
            rule = candidates[0]
            rules.append(rule)
            df = df[~((df[rule['conditions'].keys()] == rule['conditions']).all(axis=1))]
        
        self.rules = rules  # Save the rules as an instance attribute
        return rules
    
    def __repr__(self):
        repr_str = "PRISM Rules:\n"
        for i, rule in enumerate(self.rules):
            repr_str += f"Rule {i+1}: Class={rule['class']}, Support={rule['support']}, Conditions={rule['conditions']}\n"
        return repr_str




def main():
    iris = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

    prism = Prism(min_support=0.1, max_rules=10)
    prism.fit(pd.DataFrame(X_train, columns=iris.feature_names), pd.Series(y_train))

    print(prism)  # Imprime as regras geradas pelo algoritmo PRISM

    y_pred = prism.predict(pd.DataFrame(X_test, columns=iris.feature_names))

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

if __name__ == "__main__":
    main()
