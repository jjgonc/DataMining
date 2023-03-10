from scipy.stats import f_oneway  
import numpy as np
import sys
sys.path.append('../TPC1')
from tpc1 import Dataset

class F_Classif:
    def __init__(self):
        self.pvalues = None
        self.fvalues = None

    def fit(self, dataset: Dataset) -> 'F_Classif':
        # Agrupa as amostras/exemplos por classes
        groups = np.unique(dataset.getY())
        X_groups = [dataset.getX()[dataset.getY() == group] for group in groups]

        # Calcula as estatísticas F e p para cada variável do dataset em relação às diferentes classes de destino
        f_values = []
        p_values = []
        for i in range(dataset.getX().shape[1]):
            group_values = [X_group[:, i] for X_group in X_groups if X_group[:, i].size > 0]
            if len(group_values) >= 2:
                f, p = f_oneway(*group_values)
            else:
                f, p = np.nan, np.nan
            f_values.append(f)
            p_values.append(p)

        # Retorna um tuplo com os valores de F e um tuplo com os valores de p
        self.pvalues = np.array(p_values)
        self.fvalues = np.array(f_values)
        return self
    
    def transform(self, dataset: Dataset, threshold: float) -> Dataset:
        X = dataset.getX()
        features_mask = self.pvalues < threshold
        X = X[:, features_mask]
        features = np.array(dataset.getFeatureNames())[features_mask]
        return Dataset(x=X, y=dataset.getY(), feature_names=list(features), label_names=dataset.getLabelNames())
    
    def fit_transform(self, dataset: Dataset, threshold: float = 0.05) -> Dataset:
        self.fit(dataset)
        return self.transform(dataset, threshold)
    

if __name__ == "__main__":
    x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    y = np.array([10, 11, 12])

    ds = Dataset(x, y, feature_names=['a', 'b', 'c'], label_names=['d'])
    f_classif = F_Classif()
    res = f_classif.fit_transform(ds)
    print(res.getLabelNames())