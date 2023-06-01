import numpy as np
from scipy.stats import f_oneway

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

    def f_classif(self):
        class_groups = {}   # dicionario para guardar as amostras de cada classe

        # Group the samples by the classes in y
        unique_classes = np.unique(self.y)      # identificar as classes unicas na variavel y para poder percorrer cada uma delas

        for cls in unique_classes:  # percorrer cada classe
            class_groups[cls] = self.X[self.y == cls]   # guardar as amostras de cada classe no dicionario, com a classe como chave

        f_values = []
        p_values = []

        # Apply the one-way ANOVA (Analysis of Variance) test to each input feature
        for feature in range(self.X.shape[1]):      # para cada feature colecionar as amostras de cada classe e aplicar o teste ANOVA
            samples = [class_groups[cls][:, feature] for cls in unique_classes]
            f_value, p_value = f_oneway(*samples)
            f_values.append(f_value)
            p_values.append(p_value)

        return f_values, p_values


if __name__ == '__main__':
    dataset = Dataset()
    dataset.load_from_csv('dataset.csv')

    f_values, p_values = dataset.f_classif()

    print("F-values:", f_values)
    print("p-values:", p_values)
