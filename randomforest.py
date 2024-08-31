import numpy as np
from collections import Counter
from decisiontree import DecisionTree

class RandomForest:
    def __init__(self, max_depth=100, min_samples_split=2, features_count=None, trees_count=5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.features_count = features_count
        self.trees = []
        self.trees_count = trees_count

    def fit(self, X, y):
        for i in range(self.trees_count):
            tree = DecisionTree(max_depth=self.max_depth, 
                                      min_samples_split=self.min_samples_split,
                                      features_count=self.features_count
                                      )
            X_sample, y_sample = self._bootstrap(X, y)
            tree.fit(X_sample, y_sample)

            self.trees.append(tree)

    def _bootstrap(self, X, y):
        samples_count = X.shape[0]
        indexes = np.random.choice(samples_count, samples_count, replace=True)
        return X[indexes], y[indexes]
    
    def predict(self, X):
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))
        predictions = np.array(predictions)
        predictions = np.swapaxes(predictions, 0, 1)
        predictions = np.array([self._most_common_label(p) for p in predictions])
        return predictions

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]