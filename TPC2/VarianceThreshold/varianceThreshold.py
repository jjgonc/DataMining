import numpy as np

'''
Aplicar a transformação de remoção de variáveis com baixa variância, ficando apenas com as variáveis que têm uma variância acima de um determinado limiar.
'''

class VarianceThreshold:
    def __init__(self, threshold=0.0):
        self.threshold = threshold
        self.variances = None
        self.features_mask = None

    def fit(self, X):
        self.variances = np.var(X, axis=0)
        self._update_mask()  # Update the feature mask
        return self

    def transform(self, X):
        if self.features_mask is None:
            raise ValueError("Transformer has not been fitted. Call 'fit' method before 'transform'.")
        return X[:, self.features_mask]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def _update_mask(self):
        self.features_mask = np.where(self.variances > self.threshold)[0]



if __name__ == "__main__":
    # Sample input data
    # X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
    X = np.array([[1, 2, 1, 3],[5, 1, 4, 3],[0, 1, 1, 3]]) 

    # Create an instance of the VarianceThreshold transformer
    threshold = 2.0
    var_thresh = VarianceThreshold(threshold=threshold)

    # Fit the transformer to the input data
    var_thresh.fit(X)

    # Transform the input data
    transformed_X = var_thresh.transform(X)

    # Print the results
    print("Original X:")
    print(X)
    print("Transformed X:")
    print(transformed_X)
