import numpy as np

class Dataset:
    def __init__(self):
        self.X = None
        self.y = None
        self.feature_names = None
        self.label_name = None

    def getX(self):
        return self.X
    
    def getY(self):
        return self.y
    
    def getFeatureNames(self):
        return self.feature_names
    
    def getLabelNames(self):
        return self.label_name
    
    def set_data(self, X, y, feature_names, label_name):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.label_name = label_name
    
    def get_data(self):
        return self.X, self.y, self.feature_names, self.label_name
    
    def load_from_csv(self, file_path, delimiter=','):
        data = np.genfromtxt(file_path, delimiter=delimiter, skip_header=1)
        self._process_data(data)
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            self.feature_names = header_line.split(delimiter)[:-1]  # Remove a última coluna (que é a coluna da variável de saída)
            self.label_name = header_line.split(delimiter)[-1]  # Pega na última coluna (que é a coluna da variável de saída)

    def load_from_tsv(self, file_path):
        data = np.genfromtxt(file_path, delimiter='\t', skip_header=1)
        self._process_data(data)
        with open(file_path, 'r') as f:
            header_line = f.readline().strip()
            self.feature_names = header_line.split('\t')[:-1]
            self.label_name = header_line.split('\t')[-1]
    
    def _process_data(self, data):
        self.X = data[:, :-1]
        self.y = data[:, -1]
    
    def describe(self):
        # Estatísticas descritivas das variáveis de entrada (X)
        print("Descrição das variáveis de entrada:")
        print("Média:", np.mean(self.X, axis=0))  # Média
        print("Desv. Padrão:", np.std(self.X, axis=0))   # Desvio padrão
        print("Valor Min.:", np.min(self.X, axis=0))   # Valor mínimo
        print("Valor Max.:", np.max(self.X, axis=0))   # Valor máximo
        
        # Estatísticas descritivas da variável de saída (y)
        print("\nDescrição da variável de saída:")
        print("Média:", np.mean(self.y))          # Média
        print("Desv. Padrão:", np.std(self.y))           # Desvio padrão
        print("Valor Min.:", np.min(self.y))           # Valor mínimo
        print("Valor Max.:", np.max(self.y))           # Valor máximo
        print()



    def count_null_values(self):
        null_counts = np.sum(np.isnan(self.X), axis=0)
        print("Contagem de valores nulos nas variáveis de entrada:")
        print(null_counts)
    
    def replace_null_values(self, strategy='most_frequent'):
        if strategy == 'most_frequent':
            most_frequent_values = np.nanmedian(self.X, axis=0)
            self.X = np.where(np.isnan(self.X), most_frequent_values, self.X)
        elif strategy == 'mean':
            mean_values = np.nanmean(self.X, axis=0)
            self.X = np.where(np.isnan(self.X), mean_values, self.X)


def main():
    # Criar uma instância da classe Dataset
    dataset = Dataset()

    # Carregar dados do arquivo CSV
    dataset.load_from_csv('dataset.csv')

    # Obter os dados
    X, y, feature_names, label_name = dataset.get_data()

    # Imprimir os dados carregados
    print("Dados de entrada (X):")
    print(X)
    print()
    print("Dados de saída (y):")
    print(y)
    print()
    print("Nomes das features:")
    print(feature_names)
    print()
    print("Nome da label:")
    print(label_name)
    print()

    # Calcular estatísticas descritivas
    dataset.describe()

    # Contar valores nulos
    dataset.count_null_values()

    # Substituir valores nulos pelo valor mais comum
    dataset.replace_null_values(strategy='most_frequent')

    # Verificar os dados atualizados
    X, _, _, _ = dataset.get_data()
    print("Dados de entrada após substituição de valores nulos:")
    print(X)


import unittest

class DatasetTest(unittest.TestCase):
    def setUp(self):
        self.dataset = Dataset()
        self.dataset.load_from_csv('dataset.csv')
        # self.dataset.load_from_csv('dataset.tsv', delimiter='\t')

    def test_data_loading(self):
        X, y, feature_names, label_name = self.dataset.get_data()
        self.assertIsNotNone(X)
        self.assertIsNotNone(y)
        self.assertIsNotNone(feature_names)
        self.assertIsNotNone(label_name)
        self.assertEqual(len(X), 4)
        self.assertEqual(len(feature_names), 3)

    def test_describe(self):
        output = self.dataset.describe()
        # Add assertions based on the expected output

    def test_count_null_values(self):
        output = self.dataset.count_null_values()
        # Add assertions based on the expected output

    def test_replace_null_values(self):
        self.dataset.replace_null_values(strategy='most_frequent')
        X, _, _, _ = self.dataset.get_data()
        # Add assertions based on the expected output

if __name__ == '__main__':
    # unittest.main()
    main()
