import numpy as np
import matplotlib.pyplot as plt

# 1. Setup Synthetic Data
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
X_b = np.c_[np.ones((100, 1)), X]

# 2. Gradient Descent Function
def get_theta_path(X, y, theta, learning_rate, n_epochs, batch_size):
    m = len(X)
    theta_path = []
    theta_path.append(theta)
    
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        
        for i in range(0, m, batch_size):
            xi = X_shuffled[i:i+batch_size]
            yi = y_shuffled[i:i+batch_size]
            gradients = 2/batch_size * xi.T.dot(xi.dot(theta) - yi)
            theta = theta - learning_rate * gradients
            theta_path.append(theta)
            
    return np.array(theta_path)

# 3. Settings
theta_init = np.array([[2.0], [2.0]]) # Start far away to show the path clearly
# Configurations
path_sgd = get_theta_path(X_b, y, theta_init, learning_rate=0.1, n_epochs=50, batch_size=1)
path_mgd = get_theta_path(X_b, y, theta_init, learning_rate=0.1, n_epochs=50, batch_size=20)
path_bgd = get_theta_path(X_b, y, theta_init, learning_rate=0.1, n_epochs=50, batch_size=100)

# 4. Visualization (Professional Style)
plt.figure(figsize=(12, 10))

# --- A. Draw Contour (Background) ---
t1a, t1b = np.meshgrid(np.linspace(1.5, 4.5, 100), np.linspace(1.5, 4.5, 100))
Z = np.array([np.mean((4 + 3*X.flatten() - (b + w*X.flatten()))**2) 
              for b, w in zip(np.ravel(t1a), np.ravel(t1b))])
Z = Z.reshape(t1a.shape)

# Use 'coolwarm' with low alpha for readability
plt.contourf(t1a, t1b, Z, levels=30, cmap='Blues', alpha=0.3)
plt.contour(t1a, t1b, Z, levels=30, colors='k', alpha=0.1, linewidths=0.5)

# --- B. Draw Paths ---

# 1. Stochastic GD (Noise) - Thin, transparent red
plt.plot(path_sgd[:, 0], path_sgd[:, 1], "r-", linewidth=0.8, alpha=0.4, label="Stochastic GD (High Noise)")
plt.scatter(path_sgd[-1, 0], path_sgd[-1, 1], c='r', s=50, marker='x', label='SGD End')

# 2. Mini-batch GD (Balance) - Purple line
plt.plot(path_mgd[:, 0], path_mgd[:, 1], color='purple', linestyle='-', linewidth=1.5, alpha=0.8, label="Mini-batch GD (Balanced)")
plt.scatter(path_mgd[::5, 0], path_mgd[::5, 1], c='purple', s=20, alpha=0.6) # Mark every 5th step

# 3. Batch GD (Stable) - Thick Gold line
plt.plot(path_bgd[:, 0], path_bgd[:, 1], color='orange', linestyle='-', linewidth=3, label="Batch GD (Stable/Direct)")
plt.scatter(path_bgd[:, 0], path_bgd[:, 1], c='orange', s=40, edgecolors='k') # Mark every step

# --- C. Mark Start and Ideal End ---
plt.plot(theta_init[0], theta_init[1], "k^", markersize=15, label="Start Point")
plt.plot(4, 3, "r*", markersize=25, markeredgecolor='k', label="Global Minimum (Target)")

# --- D. Formatting ---
plt.xlabel(r"Bias ($\theta_0$)", fontsize=14)
plt.ylabel(r"Weight ($\theta_1$)", fontsize=14)
plt.title("Optimization Path Comparison: SGD vs Mini-batch vs Batch", fontsize=16, fontweight='bold')
plt.legend(loc="upper left", fontsize=11, framealpha=0.9, shadow=True)
plt.grid(True, alpha=0.2)
plt.xlim(1.5, 4.5)
plt.ylim(1.5, 4.5)

plt.tight_layout()
plt.savefig('gradient_descent_comparison_clean.png', dpi=300)
plt.show()