import numpy as np 
from sklearn.base import BaseEstimator


class BaseModel(BaseEstimator):
    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def predict(self, X):
        raise NotImplementedError("Subclasses should implement this method.")