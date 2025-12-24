import numpy as np 



def polynomial_features(X, degree):
    X_poly = X.copy()
    
    for i in range(2, degree + 1):
        X_pow = np.power(X, i)
        X_poly = np.hstack((X_poly, X_pow))
        
    return X_poly
