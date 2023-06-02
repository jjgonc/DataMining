import numpy as np

class NaiveBayesClassifier:
    """
    Classificador Naive Bayes Gaussiano para classificação de frases.

    Atributos:
    - classes (numpy.ndarray): array com as classes do modelo.
    - class_prob (numpy.ndarray): array com as probabilidades das classes.
    - mean: 
    - variances: variancia de cada feature para cada classe.

    Métodos:
    - fit(X_train, y_train): ajusta o modelo aos dados de treinamento.
    - predict(X_test): classifica as frases de teste usando o modelo ajustado.
    """

    def __init__(self):
        """
        Inicializa um objeto da classe NaiveBayesClassifier.
        """
        self.classes = None
        self.class_prob = None
        self.mean = None
        self.variance = None

    def fit(self, X_train, y_train):
        """
        Ajusta o modelo aos dados de treino.

        Argumentos:
        - X_train (numpy.ndarray): matriz de vetores de palavras de treinamento.
        - y_train (numpy.ndarray): array com as classes das frases de treinamento.
        """
        self.classes = np.unique(y_train)

        num_classes = len(self.classes)
        num_features = X_train.shape[1]

        self.mean = np.zeros((num_classes, num_features))
        self.variance = np.zeros((num_classes, num_features))
        self.class_prob = np.zeros(num_classes)

        for i, c in enumerate(self.classes):
            X_class = X_train[c == y_train]
            self.mean[i, :] = X_class.mean(axis=0)
            self.variance[i, :] = X_class.var(axis=0)
            self.class_prob[i] = X_class.shape[0] / X_train.shape[0] 

    def predict(self, X_train):
        """
        Classifica as frases de teste usando o modelo ajustado.

        Argumentos:
        - X_test (numpy.ndarray): matriz de vetores de palavras de teste.

        Retorna:
        - y_pred (numpy.ndarray): array com as classes preditas das frases de teste.
        """
        y_pred = []
        for x in X_train:
            class_prob = []
            for i in range(len(self.classes)):
                mean = self.mean[i]
                var = self.variance[i]
                eps = 1e-9  # small constant to avoid division by zero
                prob = np.prod(1 / np.sqrt(2 * np.pi * (var + eps)) * np.exp(-(x - mean)**2 / (2 * (var + eps))))
                class_prob.append(prob)
            y_pred.append(self.classes[np.argmax(class_prob)])
        return np.array(y_pred)   

def main():
    '''
    Chama as outras funções, cria e ajusta o classificador e calcula a acurácia do modelo na classificação das frases de teste.
    '''
    # Carregar dados
    X_train, y_train, X_test, y_test = load_data()
    
    '''
    # Dataset numérico estático
    X_train = np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6],
                    [5, 6, 7], [6, 7, 8], [7, 8, 9], [8, 9, 10]])
    y_train = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X_test = np.array([[1, 3, 5], [2, 4, 6], [3, 5, 7], [4, 6, 8]])
    y_test = np.array([0, 0, 1, 1])   
    '''

    # Criar e ajustar o classificador Naive Bayes
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    # Prever as classes das frases de teste
    y_pred = nb.predict(X_test)
    
    def accuracy(y_train, y_pred):
        accuracy = np.sum(y_train == y_pred) / len(y_train)
        return accuracy

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print('Predictions:',y_pred)

def load_data():
    """
    Carrega os dados de treino e teste, cria o vetorizador e transforma as frases em matrizes numéricas.

    Retorna:
    - X_train (numpy.ndarray): matriz de vetores de palavras de treinamento.
    - y_train (numpy.ndarray): array com as classes das frases de treinamento.
    - X_test (numpy.ndarray): matriz de vetores de palavras de teste.
    - y_test (numpy.ndarray): array com as classes das frases de teste.
    """
    # Criar um vetorizador para converter as frases em matriz numérica
    vectorizer = CountVectorizer()

    # Carregar dados
    train_sentences = ["a baixa do porto", "o mercado do bolhão é no porto", "a baixa de lisboa","o estadio é em lisboa","a rua de santa catarina fica no porto","tenho casa no porto","moro em lisboa","vivo em lisboa", "lisboa é bonita", "o porto é muito bonito", "o meu primo é do porto"]
    test_sentences = ["a câmara do porto fica no centro do porto", "o casino de lisboa","o hotel do porto","alvalade é em lisboa","aquela loja fica no porto","lisboa tem oceanario","a ponte 25 de abril fica em lisboa","vou ao porto"]

    # Ajustar o vetorizador aos dados de treinamento e transformar as frases de treinamento em matriz numérica
    X_train = vectorizer.fit_transform(train_sentences).toarray()  
    y_train = np.array([0, 0, 1, 1, 0,0,1,1,1,0,0])  # Classes correspondentes às frases de treinamento         
    # Transformar as frases de teste em matriz numérica
    X_test = vectorizer.transform(test_sentences).toarray()
    y_test = np.array([0, 1, 0, 1,0,1,1,0])
 
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer
    main()
