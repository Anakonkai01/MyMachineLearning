"""

b1 khỏi tạo trọng số cho các dữ liệu 

b2 lap T the he


khoi h(t)

sau đó m phải cập nhập trọng số của các điểm dữ liệu 

viet ham I() return 1, 0 
viet ham error() = sigma i run from 0 -> N = (w(i) * I(i) ) = 1
viet ham tinh alpha(error) = 1/2 * ln(1/error - 1)


error = tổng trọng số sai ()
# phải viết hàm trả về 0 hoặc 1 

w^t (i) = w^t*exp(-alpha^t*y*h^t)


b3 

code ham sign
H(T) = sign(sigma (alpha(t)* h(t)))
"""

import numpy as np 
from sklearn.base import clone


class AdaBoost():
    def __init__(self, stump, n_generations):
        self.stump = stump
        self.n_generations = n_generations
        
        
        
        
    def fit(self, X, y):
        # init the weight
        n_samples = X.shape[0]
        
        W = np.ones(n_samples) * (1/n_samples)
        
        # compute the error 
        
        
        
        
        
        
        
        
        