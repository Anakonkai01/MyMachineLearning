import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor

# 1. Tạo dữ liệu (Hình Parabol úp ngược)
np.random.seed(42)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - np.random.rand(16)) # Thêm nhiễu

# 2. Cấu hình Gradient Boosting
n_estimators = 50   # Tổng số cây
learning_rate = 0.1 # Tốc độ học
max_depth = 2       # Độ sâu mỗi cây (Cây yếu)

# 3. Khởi tạo F0 (Mean)
F = np.full(y.shape, np.mean(y))
models = []

# Chuẩn bị vẽ hình: Ta sẽ vẽ trạng thái tại vòng 0, 10, và 50
steps_to_plot = [0, 5, 49] # Index của các vòng
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for i in range(n_estimators):
    # a. Tính Residuals
    residuals = y - F
    
    # b. Train cây học Residuals
    tree = DecisionTreeRegressor(max_depth=max_depth)
    tree.fit(X, residuals)
    models.append(tree)
    
    # c. Cập nhật F
    update = tree.predict(X)
    F += learning_rate * update
    
    # --- VẼ HÌNH ---
    if i in steps_to_plot:
        ax_idx = steps_to_plot.index(i)
        ax = axes[ax_idx]
        
        # Vẽ dữ liệu thật
        ax.scatter(X, y, color='black', s=15, label='Data')
        
        # Vẽ đường dự đoán hiện tại
        # Để vẽ đường line mượt, ta predict trên một tập test dày đặc
        X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
        
        # Tái tạo F tại bước i cho tập test
        F_test = np.full(X_test.shape[0], np.mean(y))
        for j in range(i + 1):
            F_test += learning_rate * models[j].predict(X_test)
            
        ax.plot(X_test, F_test, color='red', linewidth=3, label=f'GBM Prediction')
        ax.set_title(f"Vòng lặp {i+1} (Tổng {i+1} cây)")
        ax.legend()
        ax.set_ylim(-1.5, 2.0)

plt.suptitle("Quá trình 'Tiến hóa' của Gradient Boosting", fontsize=16)
plt.show()