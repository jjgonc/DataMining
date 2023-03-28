import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.stats import chisquare, chi2_contingency


class Node:
    def __init__(self, splitThreshold=None, splitFeature=None, left=None, right=None, label=None, impurity=None):
        self.splitThreshold = splitThreshold
        self.splitFeature = splitFeature
        self.left = left
        self.right = right
        self.label = label
        self.impurity = impurity


class DecisionTree:
    def __init__(self, maxDepth=5, minSamplesSplit=2, minSamplesLeaf=1, maxLeafSize = None, maxFeatures=None, threshold=None, featureIdxs=None, pruneMethod=None, postPruneMethod=None, p_value=None, root=None, criterion='gini'):
        self.maxDepth = maxDepth
        self.minSamplesSplit = minSamplesSplit
        self.minSamplesLeaf = minSamplesLeaf
        self.maxLeafSize = maxLeafSize
        self.maxFeatures = maxFeatures
        self.criterion = criterion
        self.root = root
        self.pruneMethod = pruneMethod
        self.postPruneMethod = postPruneMethod
        self.threshold = threshold
        self.featureIdxs = featureIdxs
        self.p_value = p_value  # p-value usado no teste do qui-quadrado para a poda antecipada

    def fit(self, X, y):
        '''
        Serve para treinar o modelo, ou seja, construir a árvore de decisão.
        '''
        self.featureIdxs = np.arange(X.shape[1])
        self.root = self._buildTree(X, y)
        if self.pruneMethod is not None:
            self.root = self.prune_tree(self.root, X, y)
        return self

    def _buildTree(self, X, y):
        '''
        Função recursiva que constroi a árvore de decisão. 
        Retorna um nó que pode ser uma folha ou um nó de decisão, representando 
        a raiz de uma sub-árvore da árvore de decisão final.
        '''
        # verificar se o número máximo de amostras em uma folha "maxLeafSize" ou o valor p "p_value" para a poda antecipada foram definidos
        if (self.maxLeafSize is not None or self.p_value is not None):
            pre_prune = self.pre_pruning(y)
            if pre_prune:
                return Node(label=np.argmax(np.bincount(y)))

        if len(np.unique(y)) == 1:  # se só tiver uma classe
            return Node(label=y[0])
        if self.maxDepth is not None and self.maxDepth == 0: # se a profundidade for 0
            return Node(label=np.argmax(np.bincount(y)))
        if len(X) < self.minSamplesLeaf:    # se o número de amostras for menor que o número mínimo de amostras em uma folha
            return Node(label=np.argmax(np.bincount(y)))
        if len(X) < self.minSamplesSplit:   # se o número de amostras for menor que o número mínimo de amostras para dividir
            return Node(label=np.argmax(np.bincount(y)))
        if self.maxFeatures is not None and self.maxFeatures == 0: # se o número máximo de features for 0
            return Node(label=np.argmax(np.bincount(y)))
        
        # Calcula a impureza e a melhor feature para dividir, assim como o seu threshold
        newImpurity, indexOfBestFeature, bestThreshold = self._calculateCriterion(X, y)
        
        if indexOfBestFeature is None:  # se não houver melhor feature, ou seja, nenhuma divisão é possível
            return Node(label=np.argmax(np.bincount(y)))    # se não houver melhor feature, retorna a classe mais frequente
        left_indices = np.where(X[:, indexOfBestFeature] <= bestThreshold)[0].astype(int)  # indices das amostras que ficam à esquerda
        right_indices = np.where(X[:, indexOfBestFeature] > bestThreshold)[0].astype(int)  # indices das amostras que ficam à direita
        
        # caso a profundidade seja infinita, não decrementa, caso contrário decrementa porque criou uma nova folha
        if self.maxDepth is not None:
            self.maxDepth -= 1
        else:
            self.maxDepth = None
        
        left = self._buildTree(X[left_indices], y[left_indices])
        right = self._buildTree(X[right_indices], y[right_indices])

        if self.postPruneMethod == "pessimistic":
            self.pessimisticErrorPrunning(self.root, X, y)
        return Node(splitFeature=indexOfBestFeature, splitThreshold=bestThreshold, left=left, right=right, impurity=newImpurity)
    
    def _calculateCriterion(self, X, y):
        '''
        Para um determinado critério de escolha de atributos, 
        calcula o valor associado à impureza e a melhor feature para dividir, 
        assim como o seu threshold.
        '''
        if self.criterion == "entropy":
            newImpurity = self._entropy(y)
            indexOfBestFeature, bestThreshold = self.select_best_split_entropy(X, y)
        elif self.criterion == "gini_index":
            newImpurity = self._gini(y)
            indexOfBestFeature, bestThreshold = self.select_best_split_gini_index(X, y)
        elif self.criterion == "gain_ratio":
            newImpurity = self._entropy(y)
            indexOfBestFeature, bestThreshold = self.select_best_split_gain_ratio(X, y)
        else:
            raise ValueError("Invalid criterion specified")
        return newImpurity, indexOfBestFeature, bestThreshold
        
    def _gini(self, y):
        '''
        Função que calcula o índice de Gini de um conjunto de dados.
        '''
        _, counts = np.unique(y, return_counts=True)
        return 1 - np.sum((counts / len(y)) ** 2)
    
    def _entropy(self, y):
        '''
        Função que calcula a entropia de um conjunto de dados.
        '''
        _, counts = np.unique(y, return_counts=True)
        return -np.sum((counts / len(y)) * np.log(counts / len(y)))

    def _gain_ratio(self, y, left_indices, right_indices):
        '''
        Função que calcula o ganho de informação de uma divisão binária (esquerda e direita).
        Esta função calcula em separado para a esquerda e para a direita e depois faz a média ponderada.
        '''
        # Calcular a entropia do dataset completo antes da divisão
        entropy_before_split = self._entropy(y)

        # Calcular a entropia média ponderada da esquerda e da direita
        left_entropy = self._entropy(y[left_indices])
        right_entropy = self._entropy(y[right_indices])
        n_instances = len(y)
        n_left = len(left_indices)
        n_right = n_instances - n_left
        weighted_avg_entropy = (n_left / n_instances) * left_entropy + (n_right / n_instances) * right_entropy

        # Calcular a entropia de uma divisão binária (esquerda e direita)
        n_total = n_left + n_right
        p_left = n_left / n_total
        p_right = n_right / n_total
        if p_left == 0 or p_right == 0:
            split_info = 0
        else:
            split_info = - p_left * np.log2(p_left) - p_right * np.log2(p_right)  # entropy of a binary split (left and right)

        # Retornar o ganho de informação
        return (entropy_before_split - weighted_avg_entropy) / split_info if split_info != 0 else 0

    def select_best_split_entropy(self, X, y): 
        '''
        Seleção do melhor split baseado na entropia (como medida de impureza) para construção de uma árvore de decisão.
        O objetivo é encontrar a melhor divisão dos dados de entrada que maximiza a redução da impureza.

        Inicialmente calcula-se a impureza do conjunto de dados completo (y). Em seguida, para cada feature, calcula-se a impureza
        e o valor de threshold que maximiza a redução da impureza. Após dividir o conjunto de dados em duas partes (esquerda e direita),
        calcula-se a entropia condicional de Y dado X. A entropia condicional é a média ponderada da entropia de Y dado X para cada
        subconjunto de X. A entropia condicional é calculada para cada feature e threshold. A feature e o threshold que retornam a
        o maior ganho de informação são retornados. 
        '''
        indexOfBestFeature, bestThreshold, best_info_gain = None, None, -float("inf")
        entropy_Y = self._entropy(y)

        for feature_index in self.featureIdxs:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0].astype(int)
                right_indices = np.where(X[:, feature_index] > threshold)[0].astype(int)
                conditionalEntropy_YX = (len(left_indices) / len(y)) * self._entropy(y[left_indices]) + (len(right_indices) / len(y)) * self._entropy(y[right_indices])
                info_gain = entropy_Y - conditionalEntropy_YX
                if info_gain > best_info_gain:
                    indexOfBestFeature = feature_index
                    bestThreshold = threshold
                    best_info_gain = info_gain
        return indexOfBestFeature, bestThreshold

    def select_best_split_gini_index(self, X, y):
        '''
        Seleção do melhor split para uma árvore de decisão baseado no índice de Gini.
        O objetivo é encontrar a melhor divisão dos dados de entrada que maximiza a redução do índice de Gini.
        Para cada feature do conjunto de dados, são criados dois subconjuntos (esquerda e direita) para cada valor de threshold.
        Em seguida, o índice de Gini é calculado para cada subconjunto e a soma ponderada dos índices de Gini de ambos 
        os subconjuntos é calculada. A feature e o valor que resultam no menor índice de Gini ponderado são retornados como o melhor split.

        Se não for possível dividir o conjunto de dados em dois subconjuntos (por exemplo, se todos os valores da feature forem iguais), 
        o loop passa para a próxima feature. O resultado da função é o índice da melhor feature e o valor do threshold que geram a maior 
        redução do índice de Gini ponderado.
        '''
        indexOfBestFeature = None 
        bestThreshold = None 
        best_gini = - np.inf
        for feature_index in self.featureIdxs:
            for threshold in np.unique(X[:, feature_index]):
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                current_gini = (len(left_indices) / len(y)) * self._gini(y[left_indices]) \
                            + (len(right_indices) / len(y)) * self._gini(y[right_indices])
                if current_gini < best_gini:
                    indexOfBestFeature, bestThreshold, best_gini = feature_index, threshold, current_gini
        return indexOfBestFeature, bestThreshold

    def select_best_split_gain_ratio(self, X, y):
        '''
        Seleção do melhor split para uma árvore de decisão baseado no ganho de informação.
        O objetivo é encontrar a melhor divisão dos dados de entrada que maximiza o ganho de informação.
        Para cada feature do conjunto de dados, são criados dois subconjuntos (esquerda e direita) para cada valor de threshold.
        Em seguida, o ganho de informação é calculado para cada subconjunto e a soma ponderada dos ganhos de informação de ambos
        os subconjuntos é calculada. A feature e o valor que resultam no maior ganho de informação ponderado são retornados como o melhor split.

        Se não for possível dividir o conjunto de dados em dois subconjuntos (por exemplo, se todos os valores da feature forem iguais),
        o loop passa para a próxima feature. O resultado da função é o índice da melhor feature e o valor do threshold que geram a maior
        redução do índice de Gini ponderado.
        '''
        indexOfBestFeature, bestThreshold, max_gain_ratio = None, None, 0.0
        for feature_index in self.featureIdxs:
            thresholds = np.unique(X[:, feature_index])
            for threshold in thresholds:
                left_indices = np.where(X[:, feature_index] <= threshold)[0]
                right_indices = np.where(X[:, feature_index] > threshold)[0]
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                gain_ratio = self._gain_ratio(y, left_indices, right_indices)
                if gain_ratio > max_gain_ratio:
                    indexOfBestFeature, bestThreshold, max_gain_ratio = feature_index, threshold, gain_ratio
        return indexOfBestFeature, bestThreshold


    def _majorityVote(self, y, node):
        '''
        Método de resolução de conflitos para árvores de decisão.
        Se o número de amostras de uma classe for maior que o número de amostras da outra classe, a classe com maior número de amostras
        é atribuída ao nó. Caso contrário, o nó é transformado em um nó folha.
        '''
        left_label = node.left_child.label
        right_label = node.right_child.label
        if np.sum(y == left_label) > np.sum(y == right_label):
            node.left_child = None
            node.right_child = None
            node.label = left_label
        else:
            node.left_child = None
            node.right_child = None
            node.label = right_label
        return node

    def _classThreshold(self, y, node):
        '''
        Método de resolução de conflitos para árvores de decisão.
        Se o número de amostras de uma classe for maior que o número de amostras da outra classe, a classe com maior número de amostras
        é atribuída ao nó. Caso contrário, o nó é transformado em um nó folha.

        Este método é semelhante ao método _majorityVote, mas ao invés de atribuir a classe com maior número de amostras ao nó,
        ele atribui a classe com maior número de amostras se o número de amostras dessa classe for maior que o threshold.

        Parâmetros:
        y: array com as classes das amostras
        node: nó da árvore de decisão
        '''
        node_samples = len(y)
        if node.left_child.label is not None:
            left_label = node.left_child.label
            left_samples = len(np.where(y[node.left_child.indices] == left_label)[0])
        else:
            left_samples = 0
        if node.right_child.label is not None:
            right_label = node.right_child.label
            right_samples = len(np.where(y[node.right_child.indices] == right_label)[0])
        else:
            right_samples = 0
        if (left_samples / node_samples >= self.threshold) and (right_samples / node_samples >= self.threshold):
            node.left_child = None
            node.right_child = None
        elif node.left_child.label is not None and node.right_child.label is not None:
            if np.sum(y == left_label) > np.sum(y == right_label):
                node.label = left_label
            else:
                node.label = right_label
        return node

    def pre_pruning(self, y):
        '''
        Função que verifica se o split é estatisticamente significante.
        Para tal, é usado o teste do qui-quadrado, que verifica se o split é estatisticamente significante, sendp neste caso a árvore podada.

        Se o número de amostras for menor que o número máximo de folhas, a árvore é podada.
        Se o número de amostras for maior que o número máximo de folhas, mas o número de classes for igual a 1, a árvore é podada.
        Se o número de amostras for maior que o número máximo de folhas, mas o número de classes for maior que 1, a árvore não é podada.

        Parâmetros:
        y: array com as classes das amostras
        '''
        if len(y) <= self.maxLeafSize:  # verificar se o numero de amostas é menor que o maximo de folhas, significando que se pode continuar a podar
            return True
        if len(np.unique(y)) == 1 or self.maxDepth == 0:    # verificar se a profundidade maxima foi atingida
            return True
        _, p = chisquare(np.bincount(y))    # usar teste do qui-quadrado para ver se o split é estatisticamente significante
        if p > self.p_value:
            return True
        else:
            return False

    def prune_tree(self, node, X, y, threshold=None):
        '''
        Função que realiza o pruning da árvore, permitindo reduzir o overfir em modelos de árvores de decisão.

        Se o nó for uma folha, retorna o nó com a classe mais frequente.
        Se o nó não for uma folha, chama a função prune_tree para os filhos da esquerda e da direita do nó.
        Se os filhos da esquerda e da direita do nó forem folhas, verifica se o método de resolução de conflitos é o majority voting e aplica-o
        Se os filhos da esquerda e da direita do nó não forem folhas, verifica se o método de resolução de conflitos é o class threshold e aplica-o
        
        Por fim retorna o nó raiz da arvore modificada.

        Parâmetros:
        node: nó raiz da árvore
        X: array numpy com as features
        y: array numpy com as labels
        threshold: float com o threshold para o método de resolução de conflitos class threshold
        '''
        if node.label is not None:
            return node
        if self.pre_pruning(y):
            return Node(label=np.argmax(np.bincount(y)))
        node.left = self.prune_tree(node.left, X, y, threshold)
        node.right = self.prune_tree(node.right, X, y, threshold)
        if node.left.label is not None and node.right.label is not None:    # verificar se os filhos não são folhas
            if self.pruneMethod == "majority_voting":
                self._majorityVote(y, node)
        elif self.pruneMethod == "class_threshold":
            self._classThreshold(y, node)
        return node
    

    def pessimisticErrorPrunning(self, node, X, y):
        if node is None:
            return None
        if node.left is None and node.right is None:
            return node
        # Calcula a taxa de erro do nó em relação aos dados de validação
        y_pred = self.predict(X)
        error_node = 1 - accuracy_score(y, y_pred)
        # Calcula a taxa de erro esperada
        if node.left is None:
            error_expected = error_node
        else:
            y_pred_left = self.predict(X[node.left])
            error_left = 1 - accuracy_score(y[node.left], y_pred_left)
            error_expected = error_left + self.prune_tree(node.left, X[node.left], y[node.left]).impurity
        if node.right is not None:
            y_pred_right = self.predict(X[node.right])
            error_right = 1 - accuracy_score(y[node.right], y_pred_right)
            error_expected += error_right + self.prune_tree(node.right, X[node.right], y[node.right]).impurity
        else:
            error_expected = error_node
        # Se a taxa de erro esperada for maior do que a taxa de erro do nó, substitui o nó e seus filhos por um nó folha
        if error_expected > error_node:
            node.left = None
            node.right = None
            node.label = np.argmax(np.bincount(y))
        return node

        
    def predict(self, X):
        '''
        Função que faz a predição de uma amostra, para tal, chama a função _predict para cada uma das amostras, retornando um array numpy com as predictions correspondentes.

        Parâmetros:
        X: array numpy com as features das amostras a serem preditas

        Parâmetros:
        X: array numpy com as features das amostras a serem preditas
        '''
        return np.array([self._predict(x, self.root) for x in X])
    
    def _predict(self, x, node):
        '''
        Função auxiliar que percorre a árvore de decisão a partir do nó raiz 
        até chegar em um nó folha, onde a classe é atribuída. 
        
        A verificação do nó atual é feita verificando se o nó atual é uma folha. 
        
        Se for, retorna o valor do rótulo (classe) correspondente. 
        
        Caso contrário, a função verifica se o valor da feature da amostra de entrada 
        é menor que o threshold armazenado no nó atual, e segue para a subárvore esquerda 
        ou direita, dependendo do resultado da comparação. 
        
        Esse processo é repetido recursivamente até que se chegue a um nó folha.

        Parâmetros:
        x: array numpy com as features da amostra a ser predita
        node: nó raiz da árvore de decisão
        '''
        if node.label is not None:
            return node.label
        if x[node.splitFeature] < node.splitThreshold:  # verificar se o valor da feature é menor que o threshold
            return self._predict(x, node.left)
        else:
            return self._predict(x, node.right)
        
 

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
def main():
    df = pd.read_csv('weather.csv')
    label_enc = LabelEncoder()      # é necessário usar um LabelEncoder para transformar os valores categóricos em numéricos
    df['play'] = label_enc.fit_transform(df['play'])
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    dt = DecisionTree(maxDepth=3, minSamplesLeaf=2, criterion="entropy", pruneMethod="majority", postPruneMethod="pessimistic", threshold=0.1, maxLeafSize=2, p_value=0.5)
    dt.fit(X_train, y_train)       # treina o modelo
    y_pred = dt.predict(X_test)     # faz as predições
    accuracy = accuracy_score(y_test, y_pred)    # calcula a percentagem de acertos
    print("Accuracy = ", accuracy)




if __name__ == '__main__':
    main()

