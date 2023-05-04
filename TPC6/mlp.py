import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset
class MLP:
    # constructor
    # def __init__(self, dataset, X = None, Y = None):

    def predict(self, instance):
        x = np.empty([self.X.shape[1]])        
        x[0] = 1
        x[1:] = np.array(instance[:self.X.shape[1]-1])

        if self.normalized:
            x[1:] = (x[1:] - self.mu) / self.sigma 
        return np.dot(self.theta, x)
    
    def costFunction(self):
        m = self.X.shape[0]
        predictions = np.dot(self.X, self.theta)
        sqe = (predictions- self.y) ** 2
        res = np.sum(sqe) / (2*m)
        return res
    
    def buildModel():
        print("TODO build model")
    def normalize(self):
        self.mu = np.mean(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] - self.mu
        self.sigma = np.std(self.X[:,1:], axis = 0)
        self.X[:,1:] = self.X[:,1:] / self.sigma
        self.normalized = True

def test():    
    ds= Dataset("lr-example2.data")   
    
    lrmodel = MLP(ds) 
    print("Initial cost: (not optimized)", lrmodel.costFunction())
    print()
    
    print("Analytical method:")
    lrmodel.buildModel()
    print("Cost: ", lrmodel.costFunction())
    print("Coefficients:")
    lrmodel.printCoefs()     
    print("Prediction for example (3000, 3):")
    ex = np.array([3000,3,100000])
    print(lrmodel.predict(ex))

    print()
    
    print("Normalized + gradient descent:")
    lrmodel.normalize()
    print("Running GD:")
    lrmodel.gradientDescent(1000, 0.01)   
    print("FInal Cost: ", lrmodel.costFunction())
    print("Coefficients:")
    lrmodel.printCoefs()  
    print("Prediction for example (3000, 3):")
    print(lrmodel.predict(ex))

# main - tests
if __name__ == '__main__': 

    test()