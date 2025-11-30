import numpy as np
from sklearn.base import clone
from scipy.stats import mode
from .base import BaseModel
from .tree import MyDecisionTreeRegressor, MyDecisionTreeClassifier
from .linear_model import MyLogisticRegression 

# 1. BAGGING (Parallel)
class BaseBagging(BaseModel):
    def __init__(self, base_estimator, n_estimators=10):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.estimators_ = [] 

    def fit(self, X, y):
        self.estimators_ = [] 
        n_samples = X.shape[0]

        for i in range(self.n_estimators):
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_sample, y_sample = X[indices], y[indices]

            model = clone(self.base_estimator)
            model.fit(X_sample, y_sample)
            self.estimators_.append(model)
            
        print(f"Bagging: Đã huấn luyện xong {self.n_estimators} mô hình!")

class GeneralBaggingClassifier(BaseBagging):
    def predict(self, X):
        if not self.estimators_:
            raise Exception("Chưa train model!")

        predictions = np.array([model.predict(X) for model in self.estimators_])
        majority_vote = mode(predictions, axis=0)[0]
        return majority_vote.flatten()

class GeneralBaggingRegressor(BaseBagging):
    def predict(self, X):
        if not self.estimators_:
            raise Exception("Chưa train model!")

        predictions = np.array([model.predict(X) for model in self.estimators_])
        return np.mean(predictions, axis=0)


# 2. ADABOOST

class DecisionStump:
    def __init__(self):
        self.polarity = 1
        self.feature_idx = None
        self.threshold = None
        self.alpha = None

    def predict(self, X):
        n_samples = X.shape[0]
        X_column = X[:, self.feature_idx]
        predictions = np.ones(n_samples)
        
        if self.polarity == 1:
            predictions[X_column < self.threshold] = -1
        else:
            predictions[X_column < self.threshold] = 1
            predictions[X_column >= self.threshold] = -1
        return predictions

class MyAdaBoostClassifier(BaseModel):
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators_ = []

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y <= 0, -1, 1)
        w = np.full(n_samples, (1 / n_samples))
        self.estimators_ = []

        for _ in range(self.n_estimators):
            clf = DecisionStump()
            min_error = float('inf')
            
            for feature_i in range(n_features):
                X_column = X[:, feature_i]
                thresholds = np.unique(X_column)
                for threshold in thresholds:
                    for polarity in [1, -1]:
                        predictions = np.ones(n_samples)
                        if polarity == 1:
                            predictions[X_column < threshold] = -1
                        else:
                            predictions[X_column < threshold] = 1
                            predictions[X_column >= threshold] = -1

                        error = sum(w[y_ != predictions])
                        
                        if error < min_error:
                            min_error = error
                            clf.polarity = polarity
                            clf.threshold = threshold
                            clf.feature_idx = feature_i

            EPS = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - min_error + EPS) / (min_error + EPS))
            
            predictions = clf.predict(X)
            w *= np.exp(-clf.alpha * y_ * predictions)
            w /= np.sum(w)

            self.estimators_.append(clf)

    def predict(self, X):
        clf_preds = [clf.alpha * clf.predict(X) for clf in self.estimators_]
        y_pred = np.sum(clf_preds, axis=0)
        y_pred = np.sign(y_pred)
        return np.where(y_pred == -1, 0, 1)

class MyAdaBoostRegressor(BaseModel):
    def __init__(self, n_estimators=50):
        self.n_estimators = n_estimators
        self.estimators_ = [] 
        self.alphas = []

    def fit(self, X, y):
        n_samples = X.shape[0]
        w = np.full(n_samples, 1 / n_samples)
        self.estimators_ = []
        self.alphas = []
        
        for _ in range(self.n_estimators):
            tree = MyDecisionTreeRegressor(max_depth=3)
            indices = np.random.choice(n_samples, size=n_samples, replace=True, p=w)
            tree.fit(X[indices], y[indices])
            
            y_pred = tree.predict(X)
            abs_error = np.abs(y - y_pred)
            max_error = np.max(abs_error)
            if max_error == 0: break
                
            error_norm = abs_error / max_error
            avg_error = np.sum(w * error_norm)
            if avg_error >= 0.5: break
            
            beta = avg_error / (1 - avg_error)
            alpha = np.log(1/beta) if beta > 0 else 10
            w *= np.power(beta, (1 - error_norm))
            w /= np.sum(w)
            
            self.estimators_.append(tree)
            self.alphas.append(alpha)

    def predict(self, X):
        if not self.estimators_: raise Exception("Chưa train model!")
        tree_preds = np.array([model.predict(X) for model in self.estimators_])
        
        weighted_sum = np.zeros(X.shape[0])
        sum_alphas = sum(self.alphas)
        
        for i, pred in enumerate(tree_preds):
            weighted_sum += pred * self.alphas[i]
            
        return weighted_sum / sum_alphas


# 3. GRADIENT BOOSTING

class MyGradientBoostingRegressor(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = [] 
        self.initial_prediction = None 

    def fit(self, X, y):
        self.estimators_ = [] 
        self.initial_prediction = np.mean(y)
        F_current = np.full(y.shape, self.initial_prediction)
        
        for _ in range(self.n_estimators):
            residuals = y - F_current
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            residual_pred = tree.predict(X)
            F_current += self.learning_rate * residual_pred
            self.estimators_.append(tree)
            
    def predict(self, X):
        if not self.estimators_: raise Exception("Chưa train model!")
        y_pred = np.full(X.shape[0], self.initial_prediction)
        
        for tree in self.estimators_:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

class MyGradientBoostingClassifier(BaseModel):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.estimators_ = []
        self.initial_log_odds = None 

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        self.estimators_ = []
        p = np.mean(y)
        p = np.clip(p, 1e-10, 1 - 1e-10)
        self.initial_log_odds = np.log(p / (1 - p))
        F_current = np.full(y.shape, self.initial_log_odds)
        
        for _ in range(self.n_estimators):
            probabilities = self._sigmoid(F_current)
            residuals = y - probabilities
            
            tree = MyDecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            
            update_pred = tree.predict(X)
            F_current += self.learning_rate * update_pred
            self.estimators_.append(tree)

    def predict_proba(self, X):
        if not self.estimators_: raise Exception("Chưa train model!")
        F_pred = np.full(X.shape[0], self.initial_log_odds)
        # Sửa trees -> estimators_
        for tree in self.estimators_:
            F_pred += self.learning_rate * tree.predict(X)
        return self._sigmoid(F_pred)

    def predict(self, X):
        proba = self.predict_proba(X)
        return np.where(proba >= 0.5, 1, 0)


# 4. STACKING


class MyStackingClassifier(BaseModel):
    def __init__(self, estimators, final_estimator=None, blend_ratio=0.5):
        self.estimators = estimators
        self.final_estimator = final_estimator if final_estimator else MyLogisticRegression()
        self.blend_ratio = blend_ratio
        self.estimators_ = [] 

    def fit(self, X, y):
        n_samples = int(X.shape[0] * self.blend_ratio)
        X_base, y_base = X[:n_samples], y[:n_samples]
        X_meta, y_meta = X[n_samples:], y[n_samples:]
        
        self.estimators_ = []
        meta_features = [] 
        
        print(f"Stacking: Train Base Models trên {len(X_base)} mẫu...")
        for name, model in self.estimators:
            clf = clone(model)
            clf.fit(X_base, y_base)
            self.estimators_.append(clf)
            
            if hasattr(clf, "predict_proba"):
                pred_meta = clf.predict_proba(X_meta)
                if pred_meta.ndim > 1: pred_meta = pred_meta[:, 1]
            else:
                pred_meta = clf.predict(X_meta)
            meta_features.append(pred_meta)
            
        meta_X = np.column_stack(meta_features)
        
        print(f"Stacking: Train Meta Model trên {len(meta_X)} mẫu...")
        self.final_estimator.fit(meta_X, y_meta)

    def predict(self, X):
        meta_features = []
        for clf in self.estimators_:
            if hasattr(clf, "predict_proba"):
                pred = clf.predict_proba(X)
                if pred.ndim > 1: pred = pred[:, 1]
            else:
                pred = clf.predict(X)
            meta_features.append(pred)
            
        meta_X = np.column_stack(meta_features)
        return self.final_estimator.predict(meta_X)