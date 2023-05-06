import numpy as np
from dataset import Dataset


class Layer:
    def __init__(self,nodes = 2, weights=None):
        self.nodes = nodes
        self.W = weights

    def calculatePredictions(self, input):
        z2 = np.dot(self.W, input)
        a2 = np.empty([z2.shape[0] + 1])
        a2[0] = 1
        a2[1:] = sigmoid(z2)
        return a2
    
    def calculatePredictionsLastLayer(self, input):
        z3 = np.dot(self.W, input)
        return sigmoid(z3)

    def calculateCostFunction(self,input):
        Z2 = np.dot(input, self.W.T)
        A2 = np.hstack((np.ones([Z2.shape[0],1]),sigmoid(Z2)))
        return A2
    
    def calculateCostFunctionLastLayer(self, input):
        Z3 = np.dot(input, self.W.T)
        return sigmoid(Z3)
    
    
    
class DNN:
    
    def __init__(self, dataset):
        self.layers = []
        self.X, self.y = dataset.getXy()
        # colocar os 1s no X
        self.X = np.hstack ( (np.ones([self.X.shape[0],1]), self.X ) )
        

    def add(self, layer):
        self.layers.append(layer)
        

    def predict(self, instance):
        x = np.empty([len(instance)+1])        
        x[0] = 1
        x[1:] = np.array(instance[:len(instance)])

        # First Layer
        nextLayerInput = self.layers[0].calculatePredictions(x)
        
        # Layer 1 ... Layer - 1)
        for layer in self.layers[1:-1]:
            nextLayerInput = layer.calculatePredictions(nextLayerInput) 

        # Last layer
        return self.layers[-1].calculatePredictionsLastLayer(nextLayerInput)



    def costFunction(self, weights = None):
        if weights is not None:
            floorLimit = 0
            upperLimit = 0
            iter = 0
            for layer in self.layers[:-1]:
                upperLimit += (layer.nodes+1) * self.layers[iter+1].nodes
                layer.W = weights[floorLimit:upperLimit].reshape([self.layers[iter+1].nodes, layer.nodes+1])
                floorLimit = upperLimit
                iter = iter + 1
            self.layers[-1].W = weights[floorLimit:].reshape([1, self.layers[-1].nodes+1])
                    
        # First layer
        nextLayerInput = self.layers[0].calculateCostFunction(self.X)
        
        # Layer 1 ... Layer - 1)
        for layer in self.layers[1:-1]:
            nextLayerInput = layer.calculateCostFunction(nextLayerInput)

        # Last layer
        
        predictions = self.layers[-1].calculateCostFunctionLastLayer(nextLayerInput)

        m = self.X.shape[0]
        sqe = (predictions- self.y.reshape(m,1)) ** 2
        res = np.sum(sqe) / (2*m)
        return res
        
        


    def build_model(self):
        from scipy import optimize

        size = 0
        iter = 0

        for layer in self.layers[:-1]:
            size += (layer.nodes+1) * self.layers[iter+1].nodes
            iter = iter + 1

        size += self.layers[-1].nodes + 1

        initial_w = np.random.rand(size)        
        result = optimize.minimize(lambda w: self.costFunction(w), initial_w, method='BFGS', 
                                    options={"maxiter":1000, "disp":False} )
        
        weights = result.x
        floorLimit = 0
        upperLimit = 0
        iter = 0
        for layer in self.layers[:-1]:
            upperLimit += (layer.nodes+1) * self.layers[iter+1].nodes
            layer.W = weights[floorLimit:upperLimit].reshape([self.layers[iter+1].nodes, layer.nodes+1])
            floorLimit = upperLimit
            iter = iter + 1

        self.layers[-1].W = weights[floorLimit:].reshape([1, self.layers[-1].nodes+1])


def sigmoid(x):
  return 1 / (1 + np.exp(-x))



# Test different number of hidden nodes and layers
def testBuildModel():
    ds = Dataset("xnor.data")
    model = DNN(ds)
    model.add(Layer(nodes=2))
    model.add(Layer(nodes=4))
    model.add(Layer(nodes=3))
    model.build_model()

    print(" 0 0 , predict: ", model.predict(np.array([0,0]) ) )
    print(" 1 0 , predict: ", model.predict(np.array([0,1]) ) )
    print(" 1 0 , predict: ", model.predict(np.array([1,0]) ) )
    print(" 1 1 , predict: ", model.predict(np.array([1,1]) ) )


# Test XNOR with weights
def testXNOR():
    ds = Dataset("xnor.data")
    model = DNN(ds)
    model.add(Layer(nodes=2, weights=np.array([[-30,20,20],[10,-20,-20]])))
    model.add(Layer(nodes=2, weights=np.array([[-10,20,20]])))

    print(" 0 0 , predict: ", model.predict(np.array([0,0]) ) )
    print(" 1 0 , predict: ", model.predict(np.array([0,1]) ) )
    print(" 1 0 , predict: ", model.predict(np.array([1,0]) ) )
    print(" 1 1 , predict: ", model.predict(np.array([1,1]) ) )


#testXNOR()
testBuildModel()
