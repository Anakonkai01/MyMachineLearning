import numpy as np 
from sklearn.base import clone
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt 
from matplotlib.colors import ListedColormap

# data structure to store the stump
class DecisionStump():
    def __init__(self):
        self.polarity = 1 # show which direction of the tree (1 mean: left is -1, right is 1 ; -1 mean: left is 1, right is -1)
        self.feature_idx = None # which feature it is considering
        self.threshold = None # the threahold (this root which stand for if ... else ...)
        self.alpha = None # this show how much this stump STRONG
        
        

    def predict(self, X):
        n_samples = X.shape[0] # len of data points
        X_columns = X[:, self.feature_idx] # choose the current column
        
        # create predictions fill with 1 (update later)
        predictions = np.ones(n_samples)
        if self.polarity == 1:
            # left is -1, right is 1 
            predictions[X_columns < self.threshold] = -1
            # doesn't need to update the right side because it already init with 1
        else:
            predictions[X_columns < self.threshold] = 1
            predictions[X_columns >= self.threshold] = -1

        return predictions



# using decision stump only 
class AdaBoostClassifierBinary():
    def __init__(self, n_estimator):
        self.n_estimator = n_estimator
        self.stumps = [] # list to store every stump

    
    # static method
    # brute-force, find all posibilities to best threshold to minimize the weighted error
    def find_best_stump(self, X, y, w):
        n_samples, n_features = X.shape
        best_stump = DecisionStump()
        
        min_error = float('inf') # init the min is the largest
        
        # find every best threshold for each feature
        for feature_i in range(n_features):
            # take the unique value of x in feature_i 
            X_columns = X[:, feature_i]
            thresholds = np.unique(X_columns)

            # test with every stumps with each threshold and polarity to find the best threshold 
            for threshold in thresholds:
                # create a temp stump to test 
                for polarity in [1, -1]:
                    stump = DecisionStump()
                    stump.feature_idx = feature_i
                    stump.threshold = threshold
                    stump.polarity = polarity

                    # test prediction to calculate error 
                    predictions = stump.predict(X)

                    # calculate error of this temp stump 
                    # error in here is sum of weighted error 
                    error = np.sum(w[y != predictions])

                    # if this temp stump has error < min_error, store this 
                    if error < min_error:
                        min_error = error
                        best_stump = stump
        
        # return the best stump and it own error
        return best_stump, min_error



        
    def fit(self, X, y):
        # init the weight
        n_samples, n_features = X.shape

        # init W with every wi = 1/n_samples
        w = np.full(n_samples, 1/n_samples)

        # train each stump 
        for t in range(self.n_estimator):
            # step 1: train stump at t
            stump, error = self.find_best_stump(X, y, w)

            # step 2: calculate alpha (how strong is this stump)
            # alpha = 1/2 * ln(1/error - 1)
            EPS = 1e-10  # very small number to avoid divide by zero
            stump.alpha = 1/2 * np.log((1-error)/(error + EPS))

            # step 3: update the new weight for t+1
            # w_new(t+1) = w(t) * exp(- alpha * y * h(t))
            predictions = stump.predict(X) # this is h(t)
            w = w * np.exp(-stump.alpha * y * predictions)

            # standardize w to make sure the sum of all w == 1, and prepare next loop
            w /= np.sum(w)

            
            self.stumps.append(stump)

            
    def predict(self, X):
        # H(T) = sign(sum(alpha(t) * h(t)))
        stumps_pred = []
        for stump in self.stumps:
            prediction = stump.alpha *  stump.predict(X)
            stumps_pred.append(prediction)
        
        y_pred_sum = np.sum(stumps_pred, axis=0)

        find_pred = np.sign(y_pred_sum) 

        return find_pred.astype(int) # convert to int
        
        
    

# using make moon to test 
X, y = make_moons(n_samples=1000, noise=0.1, random_state=42)

# we must convert 0 to -1 because the adaboost work only with label 1 and -1
y = np.where(y == 0, -1, 1)

N_ESTIMATOR = 200
model = AdaBoostClassifierBinary(n_estimator=N_ESTIMATOR)
model.fit(X, y)

# --- 3. Hàm vẽ biên quyết định (Decision Boundary) ---
def plot_decision_boundary(model, X, y, title):
    # Tạo một lưới điểm dày đặc bao phủ toàn bộ vùng dữ liệu
    # Để hỏi mô hình dự đoán tại mọi điểm trên cái lưới này
    h = .02  # Độ mịn của lưới
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Dự đoán trên toàn bộ lưới
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Vẽ vùng màu nền (Contour plot)
    # Màu đỏ nhạt: Vùng mô hình đoán là -1
    # Màu xanh nhạt: Vùng mô hình đoán là +1
    cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.4)

    # Vẽ các điểm dữ liệu thật
    # Màu đỏ đậm: Điểm thật là -1
    # Màu xanh đậm: Điểm thật là +1
    cmap_bold = ListedColormap(['#FF0000', '#0000FF'])
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                edgecolor='k', s=40)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel("Feature 1 (X1)")
    plt.ylabel("Feature 2 (X2)")

# --- 4. Vẽ hình ---
plt.figure(figsize=(8, 6))
plot_decision_boundary(model, X, y, title=f"AdaBoost của bạn với {N_ESTIMATOR} Stumps")
plt.show()   