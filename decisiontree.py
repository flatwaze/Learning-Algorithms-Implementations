import numpy as np
import pandas as pd
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None)->None:
        self.feature = feature
        self.threshold = threshold
        self.value = value
        self.left = left
        self.right = right

    def is_leaf(self):
        return self.value is not None
    

class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100, features_count = None):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.features_count = features_count
        self.root = None
    
    def fit(self, X, y):
        self.features_count = X.shape[1] if not self.features_count else min(X.shape[1], self.features_count)
        self.root = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        samples_count, feat_count = X.shape
        labels_count = len(np.unique(y))

        if(depth >= self.max_depth or labels_count == 1 or samples_count < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value = leaf_value)
        
        #best_split
        feat_inds = np.random.choice(feat_count, self.features_count, replace=False)        
        best_feat, best_thold = self._best_split(X, y, feat_inds)

        left_inds, right_inds = self._split(X[:, best_feat], best_thold)
        left = self._grow_tree(X[left_inds, :], y[left_inds], depth+1)
        right = self._grow_tree(X[right_inds, :], y[right_inds], depth+1)
        
        return Node(best_feat, best_thold, left, right)


    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    

    def _best_split(self, X, y, feat_inds):
        best_gain = -1
        split_ind, split_thold = None, None

        for feat_ind in feat_inds:
            X_col = X[:, feat_ind]
            tholds = np.unique(X_col)

            for t in tholds:
                gain = self._information_gain(X_col, y, t)
                if gain > best_gain:
                    best_gain = gain
                    split_ind = feat_ind
                    split_thold = t

        return split_ind, split_thold
    

    #Entropy(parent) - [weighted_avg] * Entropy(child)
    def _information_gain(self, X_col, y, thold):
        #parent_entropy
        parent_entropy = self._entropy(y)

        #child_entropy
        left_inds, right_inds = self._split(X_col, thold)
        if len(left_inds) == 0 or len(right_inds) == 0:
            return 0

        n = len(y)
        n_left, n_right = len(left_inds), len(right_inds)
        e_left, e_right = self._entropy(y[left_inds]), self._entropy(y[right_inds])
        child_entropy = e_left * n_left/n + e_right * n_right/n

        return parent_entropy - child_entropy


    #Entrpy formula with freq prob
    def _entropy(self, y):
        probs = np.bincount(y)/len(y)
        ret = 0
        for p in probs:
            if p > 0:
                ret += p * np.log(p)

        return -ret
    

    def _split(self, X_col, thold):
        left_indexes = np.argwhere(X_col <= thold).flatten()
        right_indexes = np.argwhere(X_col > thold).flatten()

        return left_indexes, right_indexes
    


    def predict(self, X):
        arr = [self._trav_tree(x, self.root) for x in X]
        arr = np.array(arr)
        return arr

    def _trav_tree(self, x, node):
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._trav_tree(x, node.left)
        else:
            return self._trav_tree(x, node.right)
