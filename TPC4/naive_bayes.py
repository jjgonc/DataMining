import numpy as np

class NaiveBayesClassifier:
    """
    Classificador Naive Bayes para classificação de frases.

    Atributos:
    - classes (numpy.ndarray): array com as classes do modelo.
    - class_prob (numpy.ndarray): array com as probabilidades das classes.
    - vocab (numpy.ndarray): array com o vocabulário utilizado pelo vetorizador.

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
        self.vocab = None

    def fit(self, X_train, y_train):
        """
        Ajusta o modelo aos dados de treinamento.

        Argumentos:
        - X_train (numpy.ndarray): matriz de vetores de palavras de treinamento.
        - y_train (numpy.ndarray): array com as classes das frases de treinamento.
        """
        self.classes = np.unique(y_train)
        self.means = []
        self.variances = []
        
        for c in self.classes:
            X_class = X_train[y_train == c]
            self.means.append(np.mean(X_class, axis=0))
            self.variances.append(np.var(X_class, axis=0))

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
            for i, c in enumerate(self.classes):
                mean = self.means[i]
                variance = self.variances[i]
                eps = 1e-9  # small constant to avoid division by zero
                prob = np.prod(1 / np.sqrt(2 * np.pi * (variance + eps)) * np.exp(-(x - mean)**2 / (2 * (variance + eps))))
                class_prob.append(prob)
            y_pred.append(self.classes[np.argmax(class_prob)])
        return np.array(y_pred)


def main():
    '''
    Chama as outras funções, cria e ajusta o classificador e calcula a acurácia do modelo na classificação das frases de teste.
    '''
    # Carregar dados
    X_train, y_train, X_test, y_test = load_data()

    # Criar e ajustar o classificador Naive Bayes
    nb = NaiveBayesClassifier()
    nb.fit(X_train, y_train)

    # Prever as classes das frases de teste
    y_pred = nb.predict(X_test)
    
    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print('Predictions:',y_pred)

def load_data():
    """
    Carrega os dados de treinamento e teste, cria o vetorizador e transforma as frases em matrizes numéricas.

    Retorna:
    - X_train (numpy.ndarray): matriz de vetores de palavras de treinamento.
    - y_train (numpy.ndarray): array com as classes das frases de treinamento.
    - X_test (numpy.ndarray): matriz de vetores de palavras de teste.
    - y_test (numpy.ndarray): array com as classes das frases de teste.
    """
    # Criar um vetorizador para converter as frases em matriz numérica
    vectorizer = CountVectorizer()

    # Carregar dados
    train_sentences = ["a baixa do porto", "o mercado do bolhão é no porto", "a baixa de lisboa","o estadio é em lisboa","a rua de santa catarina fica no porto"]
    test_sentences = ["a câmara do porto fica no centro do porto", "o casino de lisboa","o hotel do porto","alvalade é em lisboa","aquela loja fica no porto"]

    # Ajustar o vetorizador aos dados de treinamento e transformar as frases de treinamento em matriz numérica
    X_train = vectorizer.fit_transform(train_sentences).toarray()  
    y_train = np.array([0, 0, 1, 1, 0])  # Classes correspondentes às frases de treinamento         
    # Transformar as frases de teste em matriz numérica
    X_test = vectorizer.transform(test_sentences).toarray()
    y_test = np.array([0, 1, 0, 1,0])
 
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer
    main()
