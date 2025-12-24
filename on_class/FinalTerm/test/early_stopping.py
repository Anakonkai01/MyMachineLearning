"""
Neural Network Implementation with Backpropagation and Early Stopping
Part 4: Improved Implementation with Early Stopping

This implementation includes:
- Multi-layer Feed-Forward Neural Network
- Backpropagation algorithm for training
- Early Stopping to prevent overfitting
- Model checkpointing
- Enhanced visualization and analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
import copy
warnings.filterwarnings('ignore')


class NeuralNetworkWithEarlyStopping:
    """
    Multi-layer Feed-Forward Neural Network with Backpropagation and Early Stopping
    
    Parameters:
    -----------
    layer_sizes : list
        List of integers specifying the number of neurons in each layer
    
    learning_rate : float
        Step size for gradient descent (default: 0.01)
    
    activation : str
        Activation function for hidden layers: 'relu', 'sigmoid', or 'tanh' (default: 'relu')
    
    patience : int
        Number of epochs to wait for improvement before stopping (default: 10)
    
    min_delta : float
        Minimum change in validation loss to qualify as improvement (default: 0.0001)
    
    restore_best_weights : bool
        Whether to restore weights from epoch with best validation loss (default: True)
    
    random_state : int
        Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu', 
                 patience=10, min_delta=0.0001, restore_best_weights=True, random_state=42):
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.activation = activation
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.random_state = random_state
        self.num_layers = len(layer_sizes)
        
        # Initialize weights and biases
        np.random.seed(random_state)
        self.weights = []
        self.biases = []
        
        # He initialization for weights (works well with ReLU)
        for i in range(len(layer_sizes) - 1):
            w = np.random.randn(layer_sizes[i+1], layer_sizes[i]) * np.sqrt(2.0 / layer_sizes[i])
            b = np.zeros((layer_sizes[i+1], 1))
            self.weights.append(w)
            self.biases.append(b)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        
        # Early stopping attributes
        self.best_val_loss = np.inf
        self.best_weights = None
        self.best_biases = None
        self.best_epoch = 0
        self.stopped_epoch = 0
        self.epochs_without_improvement = 0
        
    def _activation_function(self, z):
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, z)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(z)
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _activation_derivative(self, z):
        """Compute derivative of activation function"""
        if self.activation == 'relu':
            return (z > 0).astype(float)
        elif self.activation == 'sigmoid':
            a = self._activation_function(z)
            return a * (1 - a)
        elif self.activation == 'tanh':
            return 1 - np.tanh(z)**2
        else:
            raise ValueError(f"Unknown activation: {self.activation}")
    
    def _forward_propagation(self, X):
        """Forward propagation through the network"""
        activations = [X]
        z_values = []
        
        a = X
        for i in range(len(self.weights)):
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_values.append(z)
            
            if i == len(self.weights) - 1:
                a = z  # Linear activation for output
            else:
                a = self._activation_function(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        """Backward propagation to compute gradients"""
        m = X.shape[1]
        num_layers = len(self.weights)
        
        weight_gradients = [None] * num_layers
        bias_gradients = [None] * num_layers
        
        # Output layer error
        delta = activations[-1] - y
        
        # Backward pass
        for i in range(num_layers - 1, -1, -1):
            weight_gradients[i] = (1/m) * np.dot(delta, activations[i].T)
            bias_gradients[i] = (1/m) * np.sum(delta, axis=1, keepdims=True)
            
            if i > 0:
                delta = np.dot(self.weights[i].T, delta) * self._activation_derivative(z_values[i-1])
        
        return weight_gradients, bias_gradients
    
    def _update_parameters(self, weight_gradients, bias_gradients):
        """Update weights and biases using gradient descent"""
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
    
    def _compute_loss(self, y_true, y_pred):
        """Compute Mean Square Error loss"""
        return 0.5 * np.mean((y_true - y_pred)**2)
    
    def _save_best_weights(self):
        """Save current weights and biases as best"""
        self.best_weights = copy.deepcopy(self.weights)
        self.best_biases = copy.deepcopy(self.biases)
    
    def _restore_best_weights(self):
        """Restore the best weights and biases"""
        if self.best_weights is not None:
            self.weights = copy.deepcopy(self.best_weights)
            self.biases = copy.deepcopy(self.best_biases)
    
    def fit(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network with Early Stopping
        
        Parameters:
        -----------
        X_train : numpy array
            Training data
        y_train : numpy array
            Training labels
        X_val : numpy array
            Validation data (REQUIRED for early stopping)
        y_val : numpy array
            Validation labels (REQUIRED for early stopping)
        epochs : int
            Maximum number of training epochs
        batch_size : int
            Mini-batch size
        verbose : bool
            Whether to print progress
        """
        if X_val is None or y_val is None:
            raise ValueError("Validation data is required for early stopping!")
        
        # Transpose data
        X_train = X_train.T
        y_train = y_train.T
        X_val = X_val.T
        y_val = y_val.T
        
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        # Reset early stopping counters
        self.best_val_loss = np.inf
        self.epochs_without_improvement = 0
        self.best_epoch = 0
        
        print(f"\nTraining with Early Stopping:")
        print(f"  Patience: {self.patience} epochs")
        print(f"  Min Delta: {self.min_delta}")
        print(f"  Restore Best Weights: {self.restore_best_weights}")
        print("-" * 70)
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward and backward propagation
                activations, z_values = self._forward_propagation(X_batch)
                y_pred = activations[-1]
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                weight_grads, bias_grads = self._backward_propagation(
                    X_batch, y_batch, activations, z_values
                )
                self._update_parameters(weight_grads, bias_grads)
            
            # Compute losses
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            val_predictions = self.predict(X_val.T)
            val_loss = self._compute_loss(y_val, val_predictions.T)
            self.val_losses.append(val_loss)
            
            # Check for improvement
            if val_loss < self.best_val_loss - self.min_delta:
                # Improvement detected
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_best_weights()
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f} (Best)")
            else:
                # No improvement
                self.epochs_without_improvement += 1
                
                if verbose and (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - "
                          f"Val Loss: {val_loss:.6f} (No improvement: {self.epochs_without_improvement})")
            
            # Early stopping check
            if self.epochs_without_improvement >= self.patience:
                self.stopped_epoch = epoch
                print(f"\n{'='*70}")
                print(f"Early Stopping Triggered!")
                print(f"  Stopped at epoch: {epoch + 1}")
                print(f"  Best epoch: {self.best_epoch + 1}")
                print(f"  Best validation loss: {self.best_val_loss:.6f}")
                print(f"  Epochs without improvement: {self.epochs_without_improvement}")
                print(f"{'='*70}\n")
                break
        
        # Restore best weights if requested
        if self.restore_best_weights and self.best_weights is not None:
            self._restore_best_weights()
            if verbose:
                print(f"Restored weights from epoch {self.best_epoch + 1}\n")
    
    def predict(self, X):
        """Make predictions"""
        X = X.T
        activations, _ = self._forward_propagation(X)
        return activations[-1].T
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        predictions = self.predict(X)
        
        mse = mean_squared_error(y, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2_Score': r2
        }
    
    def plot_learning_curves(self, save_path=None):
        """Plot training and validation loss with early stopping marker"""
        plt.figure(figsize=(12, 6))
        
        epochs_range = range(1, len(self.train_losses) + 1)
        
        plt.plot(epochs_range, self.train_losses, label='Training Loss', 
                linewidth=2, color='blue', alpha=0.7)
        plt.plot(epochs_range, self.val_losses, label='Validation Loss', 
                linewidth=2, color='orange', alpha=0.7)
        
        # Mark best epoch
        if self.best_epoch < len(self.val_losses):
            plt.axvline(x=self.best_epoch + 1, color='green', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Best Epoch ({self.best_epoch + 1})')
            plt.scatter([self.best_epoch + 1], [self.val_losses[self.best_epoch]], 
                       color='green', s=200, zorder=5, marker='*', 
                       label=f'Best Val Loss: {self.best_val_loss:.6f}')
        
        # Mark stopped epoch
        if self.stopped_epoch > 0:
            plt.axvline(x=self.stopped_epoch + 1, color='red', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Stopped at Epoch {self.stopped_epoch + 1}')
        
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Learning Curves with Early Stopping', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_overfitting_analysis(self, save_path=None):
        """Plot analysis showing overfitting prevention"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        epochs_range = range(1, len(self.train_losses) + 1)
        
        # Loss curves
        axes[0].plot(epochs_range, self.train_losses, label='Training Loss', linewidth=2)
        axes[0].plot(epochs_range, self.val_losses, label='Validation Loss', linewidth=2)
        axes[0].axvline(x=self.best_epoch + 1, color='green', linestyle='--', 
                       label='Best Epoch', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss (MSE)', fontsize=12)
        axes[0].set_title('Training vs Validation Loss', fontsize=13, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Gap analysis
        gap = np.array(self.val_losses) - np.array(self.train_losses)
        axes[1].plot(epochs_range, gap, label='Val Loss - Train Loss', 
                    linewidth=2, color='purple')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        axes[1].axvline(x=self.best_epoch + 1, color='green', linestyle='--', 
                       label='Best Epoch', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Loss Gap', fontsize=12)
        axes[1].set_title('Overfitting Gap Analysis', fontsize=13, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_and_preprocess_data():
    """Load and preprocess the California Housing dataset"""
    from sklearn.datasets import fetch_california_housing
    
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {data.feature_names}")
    
    # Split data
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42
    )
    
    # Standardize
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    print(f"\nData split:")
    print(f"  Training: {X_train.shape[0]} samples")
    print(f"  Validation: {X_val.shape[0]} samples")
    print(f"  Test: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def compare_with_without_early_stopping(X_train, X_val, X_test, y_train, y_val, y_test):
    """Compare models with and without early stopping"""
    
    print("\n" + "="*70)
    print("COMPARISON: With vs Without Early Stopping")
    print("="*70)
    
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 64, 32, 16, 1]
    
    # Model WITHOUT early stopping
    print("\n1. Training model WITHOUT Early Stopping...")
    print("-" * 70)
    model_no_es = NeuralNetworkWithEarlyStopping(
        layer_sizes=layer_sizes,
        learning_rate=0.01,
        activation='relu',
        patience=1000,  # Very high patience = no early stopping
        random_state=42
    )
    model_no_es.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, verbose=False)
    
    # Model WITH early stopping
    print("\n2. Training model WITH Early Stopping...")
    print("-" * 70)
    model_with_es = NeuralNetworkWithEarlyStopping(
        layer_sizes=layer_sizes,
        learning_rate=0.05,
        activation='relu',
        patience=15,
        min_delta=0.0001,
        restore_best_weights=True,
        random_state=42
    )
    model_with_es.fit(X_train, y_train, X_val, y_val, epochs=200, batch_size=64, verbose=True)
    
    # Compare results
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    
    print("\nWithout Early Stopping:")
    print(f"  Total epochs trained: {len(model_no_es.train_losses)}")
    no_es_test = model_no_es.evaluate(X_test, y_test)
    for metric, value in no_es_test.items():
        print(f"  Test {metric}: {value:.6f}")
    
    print("\nWith Early Stopping:")
    print(f"  Total epochs trained: {len(model_with_es.train_losses)}")
    print(f"  Best epoch: {model_with_es.best_epoch + 1}")
    with_es_test = model_with_es.evaluate(X_test, y_test)
    for metric, value in with_es_test.items():
        print(f"  Test {metric}: {value:.6f}")
    
    # Calculate improvement
    print("\nImprovement with Early Stopping:")
    for metric in no_es_test.keys():
        if metric == 'R2_Score':
            improvement = ((with_es_test[metric] - no_es_test[metric]) / abs(no_es_test[metric])) * 100
            print(f"  {metric}: {improvement:+.2f}% (higher is better)")
        else:
            improvement = ((no_es_test[metric] - with_es_test[metric]) / no_es_test[metric]) * 100
            print(f"  {metric}: {improvement:+.2f}% reduction")
    
    # Visualize comparison
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Without Early Stopping
    epochs1 = range(1, len(model_no_es.train_losses) + 1)
    axes[0].plot(epochs1, model_no_es.train_losses, label='Train Loss', linewidth=2)
    axes[0].plot(epochs1, model_no_es.val_losses, label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Without Early Stopping', fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # With Early Stopping
    epochs2 = range(1, len(model_with_es.train_losses) + 1)
    axes[1].plot(epochs2, model_with_es.train_losses, label='Train Loss', linewidth=2)
    axes[1].plot(epochs2, model_with_es.val_losses, label='Val Loss', linewidth=2)
    axes[1].axvline(x=model_with_es.best_epoch + 1, color='green', linestyle='--', 
                   label='Best Epoch', linewidth=2)
    if model_with_es.stopped_epoch > 0:
        axes[1].axvline(x=model_with_es.stopped_epoch + 1, color='red', linestyle='--', 
                       label='Stopped', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (MSE)')
    axes[1].set_title('With Early Stopping', fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('comparison_early_stopping.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return model_no_es, model_with_es


def main():
    """Main function"""
    
    print("=" * 70)
    print("Neural Network with Early Stopping - Part 4")
    print("Improved Implementation with Overfitting Prevention")
    print("=" * 70)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data()
    
    # Train model with early stopping
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 64, 32, 16, 1]
    
    print(f"\nNetwork Architecture: {layer_sizes}")
    
    model = NeuralNetworkWithEarlyStopping(
        layer_sizes=layer_sizes,
        learning_rate=0.05,
        activation='relu',
        patience=15,
        min_delta=0.0001,
        restore_best_weights=True,
        random_state=42
    )
    
    model.fit(X_train, y_train, X_val, y_val, epochs=500, batch_size=64, verbose=True)
    
    # Evaluate
    print("\n" + "="*70)
    print("FINAL RESULTS")
    print("="*70)
    
    print("\nTraining Set:")
    train_metrics = model.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nValidation Set:")
    val_metrics = model.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    print("\nTest Set:")
    test_metrics = model.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Visualizations
    print("\nGenerating visualizations...")
    model.plot_learning_curves('learning_curves_with_early_stopping.png')
    model.plot_overfitting_analysis('overfitting_analysis.png')
    
    # Comparison
    print("\nRunning comparison analysis...")
    model_no_es, model_with_es = compare_with_without_early_stopping(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    
    return model


if __name__ == "__main__":
    model = main()