import numpy as np 
import matplotlib.pyplot as plt 


# this implement decision tree regresion

# Node structure to store the data 
class Node:
    def __init__(self, feature_idx = None, threshold = None, left = None, right = None, value = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left 
        self.right = right
        
        # this is for leaf node
        self.value = value # this using mean
        
        

class DecisionTreeRegressor:
    def __init__(self, min_samples_split = 2, max_depth=100):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.root = None 
        
    # calculate mean 
    def _calculate_leaf_value(self, y):
        return np.mean(y)

    def _calculate_variance(self, y):
        m = len(y)
        if m == 0:
            return 0
        return np.var(y)

    # find best split (Greedy search)
    # using weighted variance reduction
    def _get_best_split(self, X, y):
        n_samples, n_features = X.shape
        best_criteria = None  # store (feature_idx, threshold)
        best_sets = None # store (X_left, y_left, X_right, y_right)
    
        min_cost = float('inf')

        # loop each feature 
        for feature_idx in range(n_features):
            X_column = X[:, feature_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # divide the data into 2 parts 
                # create mask boolean 
                left_idxs = X_column <= threshold
                right_idxs = X_column > threshold
                
                # check if the mask is empty 
                if sum(left_idxs) == 0 or sum(right_idxs) == 0:
                    continue

                y_left = y[left_idxs]
                y_right = y[right_idxs]

                # calculate cost (weighted variance)
                val_l = self._calculate_variance(y_left)
                val_r = self._calculate_variance(y_right)

                current_cost = len(y_left) * val_l + len(y_right)*val_r

                # if this have minimum error, store it
                if current_cost < min_cost:
                    min_cost = current_cost
                    best_criteria = (feature_idx, threshold)
                    best_sets = (X[left_idxs], y[left_idxs], X[right_idxs], y[right_idxs])

        return best_criteria, best_sets 
    
    # build the tree using recursion 
    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        
        # base condition this is usually is reach the leaf
        # 1. sample too small 
        # 2. the tree is too deep
        # 3. variance = 0, does not need to split 
        if (n_samples < self.min_samples_split) or (depth >= self.max_depth) or (self._calculate_variance(y) == 0):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value)

        # find the best split at this current node
        best_criteria, best_sets = self._get_best_split(X, y)

        # if it does not find any split -> this is the leaf node 
        if best_criteria is None:
            leaf_value = self._calculate_leaf_value(y) 
            return Node(value=leaf_value)

        # split the data for recursion 
        left_X, left_y, right_X, right_y = best_sets
        feature_idx, threshold = best_criteria
        
        # recursion 
        # build left tree 
        left_subtree = self._build_tree(left_X, left_y, depth + 1)
        # build right tree
        right_subtree = self._build_tree(right_X, right_y, depth + 1) 

        return Node(feature_idx, threshold, left_subtree, right_subtree)

    # this start the build process
    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    # this predict one data point 
    def _predict_one(self, node, x):
        # recusion 
        # this is base condition, this mean it reach the leaf node
        if node.value is not None:
            return node.value 
        
        # this is not the leaf node, continue to traverse 
        if x[node.feature_idx]  <= node.threshold:
            return self._predict_one(node.left, x)
        else:
            return self._predict_one(node.right, x)

    # predict the whole data 
    def predict(self, X):
        return np.array([self._predict_one(self.root, x) for x in X])

        
        
if __name__ == "__main__":
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(80, 1), axis = 0)
    y = np.sin(X).ravel()

    # add outliers 
    y[::5] += 3 * (0.5 - np.random.rand(16))
    
    # train model 
    regressor = DecisionTreeRegressor(min_samples_split=3, max_depth=0)
    regressor.fit(X, y)

    # predict 
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_pred = regressor.predict(X_test)

    plt.scatter(X, y, s=20, edgecolors='black', c='darkorange', label='read data')
    plt.plot(X_test,y_pred, color='cornflowerblue', label='predict data')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

    

    