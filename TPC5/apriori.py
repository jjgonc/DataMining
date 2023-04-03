from typing import List, Set, Dict, Tuple
import itertools

class TransactionDataset:
    def __init__(self, transactions: List[Set[int]]):
        self.transactions = transactions
        self.freq_items = self._find_frequent_items()

    def _find_frequent_items(self):
        counts = {}
        for transaction in self.transactions:
            for item in transaction:
                # increment count for item if it already exists in item_counts, otherwise initialize count to 1
                if item in counts:
                    counts[item] += 1
                else:
                    counts[item] = 1
        # sort item_counts by count in descending order and return as a list of (item, count) tuples
        freq_items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        return freq_items

class AprioriAlgorithm:
    def __init__(self, transaction_dataset : TransactionDataset, min_support):
        self.transaction_dataset = transaction_dataset
        self.min_support = min_support
        self.frequent_itemsets = self.getFrequentItemsets()
    
    def getFrequentItemsets(self):
        """
        Generate all frequent itemsets that satisfy the minimum support threshold
        """
        candidates = self.transaction_dataset.freq_items
        freq_items = {}
        freq_itemsets = {}

        #NOTE: Passei isto para aqui porque nao faz sentido estar em baixo, uma vez que é inicializado apenas no inicio e depois atualizado com os valores resultantes da generateCandidates
        for candidate, count in candidates:
            if count >= self.min_support:    # If the candidate is frequent, add it to the list of frequent items
                freq_items[candidate] = count  # Add the candidate to the list of frequent items
            
            if not freq_items:  # in the case that there are no items that cover the min_support, finish the algorithm 
                break
                
            freq_itemsets.update(freq_items)    # Add the list of frequent items to the list of frequent itemsets. Here, each index has the list of frequent items for level k+1 (because first index is 0)
        
        k = 2
        _continue = True
        
        while _continue:
            print("k= ", k)
            candidates = self.generateCandidateItemsets(freq_itemsets, k)
            if not candidates:
                break

            # candidates = self.pruneItemsets(candidates, freq_itemsets)
            # if not candidates:
            #     break

            k += 1
        print("freq_items at the end ", freq_items)
        return freq_itemsets
    
    
    def generateCandidateItemsets(self, itemset, k):    # this method takes the frequent items of the previous level and generates the candidates for the next level
        """
        Generate candidate itemsets of length k given a frequent itemset of length k-1
        """
        candidates = itertools.combinations(itemset.keys(), k)
        candidatesList = []
        for candidate in candidates:
            candidatesList.append(candidate)
        print("l74 candidatesList: ", candidatesList)

        candidatesList = self.calculateFrequentItemset(candidatesList)
        print("l76 candidatesList: ", candidatesList)
        return candidatesList
    

    def calculateFrequentItemset(self, candidatesList):
        # for candidate in candidates:
        #     if candidate[0] not in itemset.keys() or candidate[1] not in itemset.keys():
        #         candidates.remove(candidate)
        # return candidates
        """
        Filter candidate itemsets by checking if they occur in at least one transaction in the dataset
        """
        filtered_candidates = {}
        for candidate in candidatesList:
            for transaction in self.transaction_dataset.transactions:
                if set(candidate).issubset(transaction):
                    if candidate not in filtered_candidates:
                        filtered_candidates[candidate] = 1
                    else:
                        filtered_candidates[candidate] += 1

        filteredDict = {}

        for item in filtered_candidates:
            if filtered_candidates[item] >= self.min_support:
                # print("element ", item, "removed from flitered candidates")
                filteredDict[item] = filtered_candidates[item]
        return filteredDict

    def pruneItemsets(self, candidates, itemset):
        """
        Prune candidate itemsets that contain subsets of length k-1 that are not frequent
        """
        pruned_candidates = []
        for candidate in candidates:
            subsets = self.getSubsets(candidate)
            if all([subset in itemset for subset in subsets]):
                pruned_candidates.append(candidate)
        return pruned_candidates
    
    def getSubsets(self, itemset):
        """
        Generate all non-empty subsets of an itemset
        """
        subsets = []
        for i in range(1, len(itemset)):
            subsets.extend(itertools.combinations(itemset, i))
        return subsets
    
    def generateRules(self, min_confidence):
        """
        Generate association rules from frequent itemsets
        """
        rules = []
        for itemset in self.frequent_itemsets:
            if len(itemset) > 1:
                subsets = self.getSubsets(itemset)
                for subset in subsets:
                    antecedent = frozenset(subset)
                    consequent = frozenset(set(itemset) - antecedent)
                    confidence = self.getConfidence(antecedent, consequent)
                    if confidence >= min_confidence:
                        rules.append((antecedent, consequent, confidence))
        return rules

    
    def getConfidence(self, antecedent, consequent):
        """
        Calculate the confidence of an association rule
        """
        antecedent_count = self.getCount(antecedent)
        itemset_count = self.getCount(antecedent | consequent)
        return itemset_count/antecedent_count
    
    def getCount(self, itemset):
        """
        Calculate the number of transactions that contain a given itemset
        """
        count = 0
        for transaction in self.transaction_dataset.transactions:
            if itemset.issubset(transaction):
                count += 1
        return count
    
    def printRules(self, rules):
        """
        Print the association rules generated by the algorithm
        """
        for antecedent, consequent, confidence in rules:
            print("{} => {} (Confidence: {})".format(antecedent, consequent, confidence))

if __name__ == "__main__":
    transactions = [
        {1, 2, 3},
        {1, 2, 4},
        {1, 2, 3, 4},
        {1, 3, 4},
        {1, 2, 3, 4},
        {2, 3, 4},
        {2, 3},
        {2, 3, 4},
        {2, 4},
        {3, 4}
    ]

    transactions2 = [
        {1,3,4,6},
        {2,3,5},
        {1,2,3,5},
        {1,5,6}
    ]
    
    transaction_dataset = TransactionDataset(transactions2)
    apriori = AprioriAlgorithm(transaction_dataset, 2)
    rules = apriori.generateRules(0.6)
    apriori.printRules(rules)
