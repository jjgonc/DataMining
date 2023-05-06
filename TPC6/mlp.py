import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Inicializa os pesos aleatoriamente
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def forward(self, X):
        # Propagação direta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.y_hat = self.sigmoid(self.z2)
        
        return self.y_hat
    
    def costFunction(self, X, y):
        # Cálculo da função de custo (MSE)
        self.y_hat = self.forward(X)
        J = np.sum((self.y_hat - y) ** 2) / X.shape[0]
        
        return J
    
    ''' Função de custo da linear_regression
    def costFunction(self):
        m = self.X.shape[0]
        predictions = np.dot(self.X, self.theta)
        sqe = (predictions- self.y) ** 2
        res = np.sum(sqe) / (2*m)
        return res
    '''
    
    def backward(self, X, y, learning_rate):
        # Retropropagação do erro
        delta3 = (self.y_hat - y) * self.y_hat * (1 - self.y_hat)
        dW2 = np.dot(self.a1.T, delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T) * self.a1 * (1 - self.a1)
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)
        
        # Atualização dos pesos e biases
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
    
    def buildModel(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            cost = self.costFunction(X, y)
            self.backward(X, y, learning_rate)
            if i % 1000 == 0:
                print(f"Epoch {i}: cost={cost}")

# main - tests
if __name__ == '__main__': 
    # Geração de dados de exemplo
    np.random.seed(0)
    X = np.random.randn(100, 2)
    y = np.array([int(x1*x2 > 0) for x1, x2 in X])

    # Criação da rede
    model = MLP(input_size=2, hidden_size=4, output_size=1)

    # Treino da rede
    model.buildModel(X, y, epochs=10000, learning_rate=0.1)

    # Avaliação da rede
    y_pred = model.forward(X)
    y_pred = np.round(y_pred)
    accuracy = np.mean(y_pred == y)
    print(f"Acurácia: {accuracy}")
