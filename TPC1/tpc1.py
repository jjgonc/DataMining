import numpy as np
import pandas as pd


class Dataset:
    def __init__(self, x=None, y=None, feature_names = None, label_names = None):
        self.x = x
        self.y = y
        self.feature_names = feature_names
        self.label_names = label_names

    
    def getX(self):
        return self.x
    
    def getY(self):
        return self.y
    
    def getFeatureNames(self):
        return self.feature_names
    
    def getLabelNames(self):
        return self.label_names
    
    def setX(self, x):
        self.x = x
    
    def setY(self, y):
        self.y = y
    
    def setFeatureNames(self, feature_names):
        self.feature_names = feature_names
    
    def setLabelNames(self, label_names):
        self.label_names = label_names

    def readDataset(self, filename, sep = ','):
        data = np.loadtxt(filename, delimiter = sep, dtype=str)
        self.x = data[:, :-1]    # all rows, all columns except the last one
        self.y = data[:, -1]     # all rows, only the last column
        # data = pd.read_csv(filename)
        # self.x = data.iloc[:, :-1].values
        # self.y = data.iloc[:, -1].values

    def writeDataset(self, filename, sep = ","):
        fullds = np.hstack( (self.x, self.y.reshape(len(self.y),1)))
        # np.savetxt(filename, fullds, delimiter = sep)
        pd.DataFrame(fullds).to_csv(filename, header=False, index=False)

    def nullCount(self):
        return np.sum(np.char.strip(self.x) == '') + np.sum(np.char.strip(self.y) == '')
        # return np.count_nonzero(np.isnan(self.x))

    def substituteNull(self, value):
        # self.x[np.isnan(self.x)] = value
        self.x[np.char.strip(self.x) == ''] = value





if __name__ == "__main__":
    ds = Dataset()
    ds.readDataset("notas.csv")
    print("Dataset lido com sucesso!")

    print("\nPrinting X...")
    print(ds.getX())
    print("\nPrinting Y...")
    print(ds.getY())

    print("Número de valores nulos: ", ds.nullCount())
    ds.substituteNull('000')
    ds.writeDataset("output.csv")