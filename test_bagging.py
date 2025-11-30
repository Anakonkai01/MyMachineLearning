import numpy as np 
from scipy.stats import mode 
import random 
from sklearn.base import clone # for clone the estimator 
import matplotlib.pyplot as plt

class BaseBagging():
    def __init__(self, base_estimator, n_estimator):
        self.base_estimator = base_estimator
        self.n_estimator = n_estimator
        self._estimator = []
        self.validation_sets = [] # this is a X - bootstrap(x) 36.8%

    def fit(self, X, y):
        # create new list of estimator everytime this call 
        self._estimator = []

        # bootstrap 
        n_samples = X.shape[0] # get the len of rows 
        for i in range(self.n_estimator):
            # get the indices using random.choice 
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            # store the indices of every subsets for later use
            self.validation_sets.append(indices)

            # train base estimator
            estimator = clone(self.base_estimator) 
            estimator = self.base_estimator.fit(X[indices], y[indices])

            # add the estimator 
            self._estimator.append(estimator)
            

        


# implementation of bagging
# bagging classification 
class BaggingClassifier(BaseBagging):
    def predict(self, X):
        predictions = [] 
        
        # predict X with every base estimator
        for estimator in self._estimator:
            predictions.append(estimator.predict(X)) 

        # Majority vote 
        majority_vote = mode(predictions, axis=0)[0] # using mode for classifier

        return majority_vote
         
         
# implementation of bagging
# bagging classification 
class BaggingRegressor(BaseBagging):
    def predict(self, X):
        predictions = [] 
        
        # predict X with every base estimator
        for estimator in self._estimator:
            predictions.append(estimator.predict(X)) 

        # Majority vote 
        majority_vote = np.mean(predictions, axis=0) # using mean for regressor 

        return majority_vote
 

