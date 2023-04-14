from typing import List, Set, Dict, Tuple
from itertools import chain, combinations

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
        iteractionsList = []

        for candidate, count in candidates:
            if count >= self.min_support:
                freq_items[candidate] = count 
            if not freq_items:
                break
            freq_itemsets.update(freq_items) 
        
        iteractionsList.append(freq_itemsets)
        k = 2
        _continue = True
        
        while _continue:
            candidates = self.generateCandidateItemsets(freq_itemsets, k)
            if not candidates or len(candidates) == 0:
                break

            # candidates = self.pruneItemsets(candidates, freq_itemsets)
            if not candidates:
                break
            iteractionsList.append(candidates)
            k += 1
        return iteractionsList
    
    
    def generateCandidateItemsets(self, itemset, k):    # this method takes the frequent items of the previous level and generates the candidates for the next level
        """
        Generate candidate itemsets of length k given a frequent itemset of length k-1
        """
        candidates = combinations(itemset.keys(), k)
        candidatesList = []
        for candidate in candidates:
            candidatesList.append(candidate)

        candidatesList = self.calculateFrequentItemset(candidatesList)
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

    #TODO: Falta ordenar aqui os tuplos do itemset
    def printAprioriResults(self):
        """
        Print the results of the Apriori algorithm
        """
        for i, itemset in enumerate(self.frequent_itemsets):
            myKeys = list(itemset.keys())
            myKeys.sort()
            sorted_Dict = {ind : itemset[ind] for ind in myKeys}
            print("Frequent itemsets of length %d" % (i+1))
            print(sorted_Dict)
            print()


    def powerset(self, iterable):
        s = list(iterable)
        res = chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
        return res

    def apriori_association_rules(self, freq_itemsets, min_support, min_confidence):
        # generate association rules
        for itemset in freq_itemsets:
            print("\n\nitemset: ", itemset)
            if isinstance(itemset, int):
                continue
            for antecedent in self.powerset(itemset):
                antecedent = tuple(antecedent)
                if antecedent not in freq_itemsets:
                    continue
                support_antecedent = freq_itemsets[antecedent]
                support_consequent = freq_itemsets[itemset]
                confidence = support_consequent / support_antecedent
                if confidence >= min_confidence:
                    print("{} -> {}: support = {}, confidence = {}".format(
                        list(antecedent), list(set(itemset) - set(antecedent)), support_consequent, confidence))


    def associationRules(self, min_confidence, min_support):
        """
        Generate association rules from frequent itemsets
        """
        everyItem = {}
        for itemset in self.frequent_itemsets:
            for k,v in itemset.items():
                everyItem[k] = v
        print("Pre-ordering itemset: ", everyItem)
        
        everyItem = { tuple(sorted([k] if type(k) is int else k)) : v for k, v in everyItem.items() }

        print("Post-ordering itemset: ", everyItem)
        self.apriori_association_rules(everyItem, min_support, min_confidence)


    
    
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
    apriori.printAprioriResults()

    apriori.associationRules(0.6, 2)

    # rules = apriori.generateRules(0.6)
    # apriori.printRules(rules)
