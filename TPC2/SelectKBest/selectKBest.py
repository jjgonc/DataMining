import numpy as np
from sklearn.feature_selection import f_regression

'''
Utilização da função f_regression do sklearn para calcular os scores e p-values, sendo que são filtrados os k melhores 
'''

class SelectKBest:
    def __init__(self, score_func, k):
        self.score_func = score_func
        self.k = k
        self.F = None
        self.p = None
        self.selected_features = None

    def fit(self, X, y):
        scores, p_values = self.score_func(X, y)
        self.F = scores
        self.p = p_values
        self.selected_features = np.argsort(self.p)[:self.k]    # Get the indices of the k best features

    def transform(self, X):
        if self.selected_features is None:
            raise ValueError("Transformer has not been fitted. Call 'fit' method before 'transform'.")
        return X[:, self.selected_features] # Return only the selected features

    def fit_transform(self, X, y):
        self.fit(X, y)
        return self.transform(X)


if __name__ == "__main__":
    # Sample input data
    X = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    y = np.array([0, 1, 0])


    k = 2
    select_kbest = SelectKBest(score_func=f_regression, k=k)

    select_kbest.fit(X, y)

    transformed_X = select_kbest.transform(X)

    print("Original X:")
    print(X)
    print("Transformed X:")
    print(transformed_X)
