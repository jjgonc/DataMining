import numpy as np

class NaiveBayesClassifier:
    def __init__(self):
        self.classes = None
        self.class_prob = None
        self.word_prob = None
        self.vocab = None

    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        self.class_prob = np.zeros(len(self.classes))
        self.word_prob = []
        self.vocab = self.get_vocab(X_train)
        for i in range(len(self.classes)):
            X_class = X_train[y_train == self.classes[i]]
            self.class_prob[i] = len(X_class) / len(X_train)
            words = [word for sentence in X_class for word in sentence.split()]
            word_counts = np.bincount([self.word_index(word) for word in words])
            self.word_prob.append((word_counts + 1) / (len(words) + len(self.vocab)))

    def predict(self, X_test):
        y_pred = np.zeros(X_test.shape[0])
        for i in range(X_test.shape[0]):
            sentence_words = [word for word in X_test[i].split() if word in self.vocab]
            word_indices = [self.word_index(word) for word in sentence_words]
            class_probs = np.zeros(len(self.classes))
            for j in range(len(self.classes)):
                # print(len(self.class_prob),len(self.word_prob))
                print(word_indices)
                class_probs[j] = self.class_prob[j] * np.prod(self.word_prob[j][word_indices])
            y_pred[i] = self.classes[np.argmax(class_probs)]
        return y_pred


    def get_vocab(self, X_train):
        '''
        Helper method that creates a vocabulary of unique words from a list of sentences.
        The classifier needs a vocabulary to represent each word as a number to perform calculations efficiently.

        Parameters:
            sentences : list, a list of sentences.
        Returns:
            vocab : list, a sorted list of unique words.
        '''
        vocab_set = set()
        for sentence in X_train:
            words = sentence.split(sentence)
            for word in words:
                vocab_set.add(word)
        vocab_list = list(vocab_set)
        return vocab_list


    def word_index(self, word):
        '''
        Helper method that maps a word to its index in the vocabulary.
        The classifier needs to represent each word as a number to perform calculations efficiently

        Parameters:
        word : str, the word to be mapped to its index in the vocabulary.
        Returns:
        index : int, the index of the word in the vocabulary.
        '''
        if word in self.vocab:
            index = self.vocab.index(word)
        else:
            self.vocab.append(word)
            index = len(self.vocab) - 1
        return index

def main():
    # Load data
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

def load_data():
    # Criar um vetorizador para converter as frases em matriz numérica
    vectorizer = CountVectorizer()

    # Load data from files or elsewhere
    train_sentences = ["this is a sentence", "this is another sentence", "yet another sentence"]
    test_sentences = ["this is a test", "another test sentence"]

    # Ajustar o vetorizador aos dados de treinamento e transformar as frases de treinamento em matriz numérica
    X_train = vectorizer.fit_transform(train_sentences).toarray()  # FIXME - Mudar as frases 
    y_train = np.array([0, 1, 1])  # Classes correspondentes às frases de treinamento          # FIXME - Mudar as labels/classes
    # Transformar as frases de teste em matriz numérica
    X_test = vectorizer.transform(test_sentences).toarray()
    y_test = np.array([0, 1])
 
    return X_train, y_train, X_test, y_test

if __name__ == "__main__":
    from sklearn.metrics import accuracy_score
    from sklearn.feature_extraction.text import CountVectorizer
    main()
