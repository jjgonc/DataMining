import numpy as np
# import pandas as pd


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
    
    def getFeatureNames(self, filename):
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
        # data = np.loadtxt(filename, delimiter = sep, dtype=str)
        data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype='str')
        features = np.genfromtxt(filename, delimiter=',', max_rows=1, dtype='str')
        self.x = data[:, :-1]    # all rows, all columns except the last one
        self.y = data[:, -1]     # all rows, only the last column
        self.feature_names = features
        # data = pd.read_csv(filename)
        # self.x = data.iloc[:, :-1].values
        # self.y = data.iloc[:, -1].values

    def writeDataset(self, filename, sep = ","):
        fullds = np.hstack( (self.x, self.y.reshape(len(self.y),1)))
        fullds = np.vstack( (self.feature_names, fullds))
        # pd.DataFrame(fullds).to_csv(filename, header=False, index=False)
        np.savetxt(filename, fullds, fmt="%s", delimiter=',')

    def nullCount(self):
        return np.sum(np.char.strip(self.x) == '') + np.sum(np.char.strip(self.y) == '')
        # return np.count_nonzero(np.isnan(self.x))

    def substituteNullWithMean(self):
        data = self.x
        marks = np.delete(data, 0, 1)   # remove the first column that contains the student's name
        marks_asArray = np.zeros((len(marks), len(marks[0])))   # create a new array with the same size as the marks array
        for l in range(len(marks)):    # transform the marks array into a float array with the missing str values replaced by np.nan
            for c in range(len(marks[l])):
                if marks[l][c] == '':
                    marks_asArray[l][c] = np.nan
                else:
                    marks_asArray[l][c] = float(marks[l][c])

        medias = np.round(np.nanmean(marks_asArray, axis=0), decimals=1)  # calculate the mean of each column
        marks_asArray[np.isnan(marks_asArray)] = np.take(medias, np.isnan(marks_asArray).nonzero()[1])  # replace the np.nan values with the mean of the column
        
        marks_asArray = np.hstack((data[:, 0].reshape(len(data), 1), marks_asArray))  # add the student's name to the array
        self.x = marks_asArray





if __name__ == "__main__":
    ds = Dataset()
    ds.readDataset("notas.csv")
    print("Dataset lido com sucesso!")

    print("\nPrinting feature names...")
    print(ds.feature_names)
    print("\nPrinting X...")
    print(ds.getX())
    print("\nPrinting Y...")
    print(ds.getY())

    print("Número de valores nulos: ", ds.nullCount())
    ds.substituteNullWithMean()
    ds.writeDataset("output.csv")