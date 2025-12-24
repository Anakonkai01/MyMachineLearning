import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor


class MyGradientBoostingRegression:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        
        # list of trees 
        self.trees = []
        self.initial_prediction = None
        
    def fit(self, X, y):
        # 1. init F0(y) = mean(y)
        self.initial_prediction = np.mean(y)

        # create an the current prediction 
        y_pred = np.full(len(y), self.initial_prediction)

        # 2. boosting
        for _ in range(self.n_estimators):
            # 2.1 calculate residuals (negative gradient)
            residuals = (y - y_pred)

            # 2.2 train tree to learn this residual
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals) # model learn to predict the residuals 
            
            # 2.3 update new prediction for next loop
            # F_new = F_old + learning_rate * h(x)
            update = tree.predict(X)
            y_pred += self.learning_rate*update
            
            # 2.4 store the tree 
            self.trees.append(tree)

    def predict(self, X):
        # 1. start predict with mean y 
        y_pred = np.full(len(X), self.initial_prediction)

        # 2. sum of all predictions of each tree 
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)

        return y_pred
    
        
      
      
if __name__ == "__main__":
    # 1. Tạo dữ liệu hình sin nhiễu (Non-linear)
    np.random.seed(42)
    X = np.sort(5 * np.random.rand(80, 1), axis=0)
    y = np.sin(X).ravel()
    y[::5] += 3 * (0.5 - np.random.rand(16)) # Thêm nhiễu

    # 2. Huấn luyện Gradient Boosting
    # Thử thay đổi n_estimators để thấy sự tiến hóa
    # 1 cây -> Giống cái cây hôm qua
    # 100 cây -> Đường cong mượt mà
    model = MyGradientBoostingRegression(n_estimators=1, learning_rate=0.1, max_depth=1)
    model.fit(X, y)

    # 3. Dự đoán
    X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
    y_pred = model.predict(X_test)

    # 4. Vẽ hình
    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="orange", label="Data", s=20, edgecolor="black")
    plt.plot(X_test, y_pred, color="blue", linewidth=2, label="GBM Prediction")
    plt.title("Gradient Boosting Regressor (MSE) from Scratch")
    plt.legend()
    plt.show()

    
    
    