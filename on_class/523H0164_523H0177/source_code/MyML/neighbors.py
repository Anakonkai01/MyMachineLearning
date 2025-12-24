from .base import BaseModel
import numpy as np
from collections import Counter # Dùng để vote


class MyKNNClassifier(BaseModel):
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict_one(x) for x in X]
        return np.array(predicted_labels)

    def _predict_one(self, x):        
        # Tính khoảng cách từ x đến TẤT CẢ các điểm trong X_train
        # Dùng công thức Euclid: sqrt(sum((x1-x2)^2))
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        
        # Tìm K người hàng xóm gần nhất
        k_indices = np.argsort(distances)[:self.k]
        
        #  Lấy nhãn (label) của k người hàng xóm đó
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        
        #(Majority Vote)
        most_common = Counter(k_nearest_labels).most_common(1)
        
        return most_common[0][0]
    



class MyKNNRegressor(MyKNNClassifier): # Kế thừa lại logic tính khoảng cách
    def _predict_one(self, x):
        distances = np.sqrt(np.sum((self.X_train - x)**2, axis=1))
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices] # Ở đây là giá trị thực (float)

        # Thay vì bầu cử, ta tính trung bình
        return np.mean(k_nearest_labels)