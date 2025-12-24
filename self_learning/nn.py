import numpy as np 
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


# load the data 
def load_data():
    digits = load_digits()
    X = digits.data / 16.0 # standards to 0-1
    y = digits.target.reshape(-1,1) # ? -1,1

    encoder = OneHotEncoder(sparse_output=False) # ?: what is spare output
    y_onehot = encoder.fit_transform(y)
    
    
    # split train test 
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot)

    # tranpose to have (feature, shape) -> (64, m)
    return X_train.T, X_test.T, y_train.T, y_test.T




# load the data 
X_train, X_test, y_train, y_test = load_data()


class MyNeuralNet:
    def __init__(self, input_size, hidden_size, output_size):
        # init random weight bias
        self.W1 = np.random.randn(hidden_size, input_size) * 0.01 # using random from gauss distribution
        self.b1 = np.zeros((hidden_size,1))
        self.W2 = np.random.randn((output_size,hidden_size)) * 0.01
        self.b2 = np.zeros((output_size,1))

    def relu(self, Z):
        return np.maximum(0, Z) # use maximum not max


    """
        soft max for output layer, why we need minus max, because python tran so ...
    """
    def softmax(self, Z):
        exp_Z = np.exp(Z - np.max(Z, axis=0, keepdims=True))
        return exp_Z / np.sum(exp_Z, axis=0, keepdims=True)
    
    
    def forward(self, X):
        
        Z1 = np.dot(self.W1, X) + self.b1
        A1 = self.relu(Z1)
        Z2 = np.dot(self.W2, A1) + self.b2 
        A2 = self.softmax(Z2)

        # store cache for backpropagation
        self.cache = (X, Z1, A1, Z2, A2)
        return A2 
        
    def backward(self, X, y, learning_rate=0.1)
    