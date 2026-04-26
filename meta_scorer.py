import numpy as np
import dill as pickle  # Using dill to match your RCF model serialization
import os
from sklearn.preprocessing import StandardScaler

class CostSensitiveMetaLearner:
    """
    Production-Grade Logistic Regression Meta-Learner.
    Features: L2 Regularization, Persistent Internal Scaler, Early Stopping,
              Input Validation, and Disk Persistence.
    """
    def __init__(self, learning_rate=0.1, epochs=5000, cost_fn=10, cost_fp=2, lambda_reg=0.1, epsilon=1e-5):
        self.lr = learning_rate
        self.epochs = epochs
        self.cost_fn = cost_fn
        self.cost_fp = cost_fp
        self.lambda_reg = lambda_reg  
        self.epsilon = epsilon        
        
        self.weights = None
        self.bias = None
        
        # ISSUE 5 FIX: Use a persistent scaler object to prevent Normalization Leakage
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _validate_input(self, X):
        """Ensures the input matrix perfectly matches the pipeline expectations."""
        if X.shape[1] != 2:
            raise ValueError(f"[ERROR] Meta-Learner expects exactly 2 features (CatBoost, RCF). Received: {X.shape[1]}")

    def fit(self, X, y):
        self._validate_input(X)
        
        # 1. Normalization (Learns from training data only)
        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        y = np.array(y)

        # 2. Map SOC Penalties directly into the gradients
        sample_weights = np.where(y == 1, self.cost_fn, self.cost_fp)

        for epoch in range(self.epochs):
            linear_model = np.dot(X_scaled, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            weighted_error = (y_predicted - y) * sample_weights

            # 3. Calculate Gradients with L2 Penalty
            dw = (1 / n_samples) * np.dot(X_scaled.T, weighted_error) + (self.lambda_reg / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(weighted_error)

            # 4. Convergence Monitoring
            if np.max(np.abs(dw)) < self.epsilon and abs(db) < self.epsilon:
                print(f"      ↳ Convergence reached at epoch {epoch}.")
                break

            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
        self._is_fitted = True

    def predict_proba(self, X):
        if not self._is_fitted:
            raise RuntimeError("[ERROR] Meta-Learner must be fitted before predicting.")
        self._validate_input(X)
        
        # ISSUE 5 FIX: Strictly .transform() to prevent testing data variance from leaking
        X_scaled = self.scaler.transform(X)
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        return self._sigmoid(linear_model)

    # --- PERSISTENCE METHODS ---
    def save_model(self, filepath="Saves/meta_learner.pkl"):
        """Serializes the trained model and its scaler to disk."""
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
            
        # Ensure the Saves directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        state = {
            'weights': self.weights,
            'bias': self.bias,
            'scaler': self.scaler,  # Save the scaler state mathematically frozen
            'params': {
                'learning_rate': self.lr,
                'epochs': self.epochs,
                'cost_fn': self.cost_fn,
                'cost_fp': self.cost_fp,
                'lambda_reg': self.lambda_reg,
                'epsilon': self.epsilon
            }
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"💾 Model state successfully saved to {filepath}")

    @classmethod
    def load_model(cls, filepath="Saves/meta_learner.pkl"):
        """Loads a pre-trained model and its scaler from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
            
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            
        # Reconstruct the object
        instance = cls(**state['params'])
        instance.weights = state['weights']
        instance.bias = state['bias']
        instance.scaler = state['scaler']  # Restore the frozen scaler
        instance._is_fitted = True
        
        print(f"Model state successfully loaded from {filepath}")
        return instance


def train_fusion_meta_learner(X_train_meta, y_train, COST_FN=10, COST_FP=2, lambda_reg=0.1):
    print(f"\nTraining Meta-Learner (FN Penalty: {COST_FN}, FP Penalty: {COST_FP}, L2: {lambda_reg})...")
    
    meta_model = CostSensitiveMetaLearner(
        learning_rate=0.1, 
        epochs=5000, 
        cost_fn=COST_FN, 
        cost_fp=COST_FP,
        lambda_reg=lambda_reg,
        epsilon=1e-5
    )
    meta_model.fit(X_train_meta, y_train)
    
    print("Meta-Learner Training Complete!")
    return meta_model