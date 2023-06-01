import numpy as np
from scipy.stats import f_oneway


'''
Aplicar regressão linear a cada feature e guardar o p-value correspondente
'''


class Dataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_name = None

    def set_data(self, X, y, feature_names, label_name):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.label_name = label_name

    def get_data(self):
        return self.X, self.y, self.feature_names, self.label_name

    def load_from_csv(self, file_path, delimiter=','):
        data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=1)
        self.X = data[:, :-1]
        self.y = data[:, -1]
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            self.feature_names = header_line.split(delimiter)[:-1]
            self.label_name = header_line.split(delimiter)[-1]

    def load_from_tsv(self, file_path):
        data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
        self.X = data[:, :-1]
        self.y = data[:, -1]
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            self.feature_names = header_line.split('\t')[:-1]
            self.label_name = header_line.split('\t')[-1]


from sklearn.linear_model import LinearRegression

class F_Regression:
    def __init__(self):
        self.pvalues = None

    def fit(self, dataset):
        X, y, _, _ = dataset.get_data()
        self.pvalues = []

        # Apply linear regression to each feature
        for feature in range(X.shape[1]):
            model = LinearRegression()
            model.fit(X[:, feature].reshape(-1, 1), y)  # passar o x e o y e aplicar a regressao linear
            pvalue = model.score(X[:, feature].reshape(-1, 1), y)   # obter o pvalue
            self.pvalues.append(pvalue)

        return self

    # def transform(self, dataset, threshold):
    #     X, y, feature_names, label_name = dataset.get_data()
    #     features_mask = np.array(self.pvalues) < threshold
    #     X = X[:, features_mask]
    #     feature_names = np.array(feature_names)[features_mask]
    #     transformed_dataset = Dataset()
    #     transformed_dataset.set_data(X, y, feature_names, label_name)
    #     return transformed_dataset

    # def fit_transform(self, dataset, threshold=0.05):
    #     self.fit(dataset)
    #     return self.transform(dataset, threshold)
    


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_from_csv('dataset.csv')

    f_regression = F_Regression()
    f_regression.fit(dataset)
    print("P-values:", f_regression.pvalues)

    # transformed_dataset = f_regression.transform(dataset, threshold=0.05)
    
