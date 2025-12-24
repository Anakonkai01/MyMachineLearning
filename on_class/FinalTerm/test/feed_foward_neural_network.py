"""
Neural Network Implementation with Backpropagation Algorithm
Part 2: Implementation without Early Stopping

This implementation includes:
- Multi-layer Feed-Forward Neural Network
- Backpropagation algorithm for training
- Mean Square Error loss function
- Application to UCI Housing dataset (regression problem)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


class NeuralNetwork:
    """
    Multi-layer Feed-Forward Neural Network with Backpropagation
    
    Parameters:
    -----------
    layer_sizes : list
        List of integers specifying the number of neurons in each layer
        Example: [10, 64, 32, 1] means input=10, hidden1=64, hidden2=32, output=1
    
    learning_rate : float
        Step size for gradient descent (default: 0.01)
    
    activation : str
        Activation function for hidden layers: 'relu', 'sigmoid', or 'tanh' (default: 'relu')
    
    random_state : int
        Random seed for reproducibility (default: 42)
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, activation='relu', random_state=42):
        self.layer_sizes = layer_sizes # list that store all the layer, include input, hidden, output layers
        self.learning_rate = learning_rate
        self.activation = activation # options for activation function, default is relu
        self.random_state = random_state
        self.num_layers = len(layer_sizes) # including input and output layers
        
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
        """
        Forward propagation through the network
        
        Parameters:
        -----------
        X : numpy array of shape (n_features, n_samples)
            Input data
        
        Returns:
        --------
        activations : list
            List of activations for each layer (including input)
        z_values : list
            List of pre-activation values for each layer
        """
        activations = [X]
        z_values = []
        
        a = X
        for i in range(len(self.weights)):
            # Linear transformation: z = W * a + b
            z = np.dot(self.weights[i], a) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            if i == len(self.weights) - 1:
                # Output layer: use linear activation for regression
                a = z
            else:
                # Hidden layers: use specified activation
                a = self._activation_function(z)
            
            activations.append(a)
        
        return activations, z_values
    
    def _backward_propagation(self, X, y, activations, z_values):
        """
        Backward propagation to compute gradients
        
        Parameters:
        -----------
        X : numpy array
            Input data
        y : numpy array
            True labels
        activations : list
            Activations from forward pass
        z_values : list
            Pre-activation values from forward pass
        
        Returns:
        --------
        weight_gradients : list
            Gradients for weights
        bias_gradients : list
            Gradients for biases
        """
        m = X.shape[1]  # number of samples
        num_layers = len(self.weights)
        
        weight_gradients = [None] * num_layers
        bias_gradients = [None] * num_layers
        
        # Output layer error (delta)
        # For MSE loss with linear activation: δ = (ŷ - y)
        delta = activations[-1] - y
        
        # Backward pass through layers
        for i in range(num_layers - 1, -1, -1):
            # Compute gradients
            weight_gradients[i] = (1/m) * np.dot(delta, activations[i].T)
            bias_gradients[i] = (1/m) * np.sum(delta, axis=1, keepdims=True)
            
            if i > 0:
                # Propagate error to previous layer
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
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100, batch_size=32, verbose=True):
        """
        Train the neural network using backpropagation
        
        Parameters:
        -----------
        X_train : numpy array of shape (n_samples, n_features)
            Training data
        y_train : numpy array of shape (n_samples, n_outputs)
            Training labels
        X_val : numpy array (optional)
            Validation data
        y_val : numpy array (optional)
            Validation labels
        epochs : int
            Number of training epochs
        batch_size : int
            Mini-batch size for training
        verbose : bool
            Whether to print training progress
        """
        # Transpose data to shape (n_features, n_samples)
        X_train = X_train.T
        y_train = y_train.T
        
        if X_val is not None and y_val is not None:
            X_val = X_val.T
            y_val = y_val.T
        
        n_samples = X_train.shape[1]
        n_batches = int(np.ceil(n_samples / batch_size))
        
        for epoch in range(epochs):
            # Shuffle training data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[:, indices]
            y_shuffled = y_train[:, indices]
            
            epoch_loss = 0
            
            # Mini-batch gradient descent
            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = min((batch + 1) * batch_size, n_samples)
                
                X_batch = X_shuffled[:, start_idx:end_idx]
                y_batch = y_shuffled[:, start_idx:end_idx]
                
                # Forward propagation
                activations, z_values = self._forward_propagation(X_batch)
                y_pred = activations[-1]
                
                # Compute loss
                batch_loss = self._compute_loss(y_batch, y_pred)
                epoch_loss += batch_loss
                
                # Backward propagation
                weight_grads, bias_grads = self._backward_propagation(
                    X_batch, y_batch, activations, z_values
                )
                
                # Update parameters
                self._update_parameters(weight_grads, bias_grads)
            
            # Average epoch loss
            avg_train_loss = epoch_loss / n_batches
            self.train_losses.append(avg_train_loss)
            
            # Compute validation loss if validation data provided
            if X_val is not None and y_val is not None:
                val_predictions = self.predict(X_val.T)
                val_loss = self._compute_loss(y_val, val_predictions.T)
                self.val_losses.append(val_loss)
            
            # Print progress
            if verbose and (epoch + 1) % 10 == 0:
                if X_val is not None and y_val is not None:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f} - Val Loss: {val_loss:.6f}")
                else:
                    print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.6f}")
    
    def predict(self, X):
        """
        Make predictions on new data
        
        Parameters:
        -----------
        X : numpy array of shape (n_samples, n_features)
            Input data
        
        Returns:
        --------
        predictions : numpy array of shape (n_samples, n_outputs)
            Predicted values
        """
        X = X.T
        activations, _ = self._forward_propagation(X)
        return activations[-1].T
    
    def evaluate(self, X, y):
        """
        Evaluate model performance
        
        Returns:
        --------
        metrics : dict
            Dictionary containing MSE, RMSE, MAE, and R² score
        """
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
        """Plot training and validation loss curves"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Training Loss', linewidth=2)
        if self.val_losses:
            plt.plot(self.val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Learning Curves', fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def load_and_preprocess_data():
    """
    Load and preprocess the California Housing dataset
    This is a well-known regression dataset from sklearn
    
    Returns:
    --------
    X_train, X_val, X_test, y_train, y_val, y_test, scaler
    """
    from sklearn.datasets import fetch_california_housing
    
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    print(f"Dataset shape: {X.shape}")
    print(f"Features: {data.feature_names}")
    print(f"Target: Median house value (in $100,000s)")
    
    # Split data: 70% train, 15% validation, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # 0.176 * 0.85 ≈ 0.15
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    # Reshape targets
    y_train = y_train.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)
    
    print(f"\nData split:")
    print(f"  Training set: {X_train.shape[0]} samples")
    print(f"  Validation set: {X_val.shape[0]} samples")
    print(f"  Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def main():
    """Main function to demonstrate the neural network"""
    
    print("=" * 70)
    print("Neural Network with Backpropagation - Part 2")
    print("Regression on California Housing Dataset")
    print("=" * 70)
    print()
    
    # Load and preprocess data
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = load_and_preprocess_data()
    
    # Define network architecture
    input_size = X_train.shape[1]
    layer_sizes = [input_size, 64, 32, 16, 1]  # Multi-layer network
    
    print(f"\nNetwork Architecture:")
    print(f"  Input layer: {layer_sizes[0]} neurons")
    for i, size in enumerate(layer_sizes[1:-1], 1):
        print(f"  Hidden layer {i}: {size} neurons (ReLU activation)")
    print(f"  Output layer: {layer_sizes[-1]} neuron (Linear activation)")
    print()
    
    # Initialize neural network
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        learning_rate=0.001,
        activation='relu',
        random_state=42
    )
    
    # Train the model
    print("Training the model...")
    print("-" * 70)
    model.fit(
        X_train, y_train,
        X_val, y_val,
        epochs=200,
        batch_size=64,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    
    # Evaluate on training set
    print("\nTraining Set Performance:")
    train_metrics = model.evaluate(X_train, y_train)
    for metric, value in train_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Evaluate on validation set
    print("\nValidation Set Performance:")
    val_metrics = model.evaluate(X_val, y_val)
    for metric, value in val_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Evaluate on test set
    print("\nTest Set Performance:")
    test_metrics = model.evaluate(X_test, y_test)
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.6f}")
    
    # Plot learning curves
    print("\nPlotting learning curves...")
    model.plot_learning_curves(save_path='learning_curves_part2.png')
    
    # Make sample predictions
    print("\nSample Predictions (first 10 test samples):")
    print("-" * 70)
    predictions = model.predict(X_test[:10])
    print(f"{'True Value':<15} {'Predicted Value':<18} {'Error':<10}")
    print("-" * 70)
    for true, pred in zip(y_test[:10], predictions):
        error = abs(true[0] - pred[0])
        print(f"{true[0]:<15.4f} {pred[0]:<18.4f} {error:<10.4f}")
    
    print("\n" + "=" * 70)
    print("Analysis Complete!")
    print("=" * 70)
    
    return model, (X_train, X_val, X_test, y_train, y_val, y_test)


if __name__ == "__main__":
    model, data = main()