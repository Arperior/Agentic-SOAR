import numpy as np
import dill as pickle  
import os
from sklearn.preprocessing import StandardScaler

class CostSensitiveMetaLearner:
    """
    Production-Grade Logistic Regression Meta-Learner.
    Features: L2 Regularization, Persistent Internal Scaler, Early Stopping,
              Input Validation, and Disk Persistence.

    Now accepts 3 meta-features:
      [0] CatBoost binary probability
      [1] RCF anomaly score
      [2] Incident agent uncertainty (entropy over attack-class softmax)
    The incident agent's per-class confidence distribution carries signal about
    rare attack types that the binary CatBoost score alone underweights. Rather
    than feeding all N class probabilities (which would overfit on a 2-weight
    logistic model), we reduce to a single entropy scalar: high entropy = the
    incident agent is uncertain = borderline traffic the meta-learner should
    scrutinise more carefully.
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
        
        self.scaler = StandardScaler()
        self._is_fitted = False

    def _sigmoid(self, z):
        z = np.clip(z, -250, 250)
        return 1.0 / (1.0 + np.exp(-z))

    def _validate_input(self, X):
        """Ensures the input matrix perfectly matches the pipeline expectations."""
        if X.shape[1] != 4:
            raise ValueError(
                f"[ERROR] Meta-Learner expects exactly 4 features "
                f"(CatBoost prob, RCF score, Incident entropy, Top-2 margin). Received: {X.shape[1]}"
            )

    def fit(self, X, y):
        self._validate_input(X)
        
        # 1. Normalization (Learns from training data only)
        X_scaled = self.scaler.fit_transform(X)

        n_samples, n_features = X_scaled.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0   # float from the start; avoids numpy.float64 in saved state
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
        
        X_scaled = self.scaler.transform(X)
        linear_model = np.dot(X_scaled, self.weights) + self.bias
        raw_prob = self._sigmoid(linear_model)
        
        # FIX (Section 2, Issue 2): Clip the floating point to strictly [0.0, 1.0]
        # This prevents edge-case inputs from outputting 1.0000000000000002
        return np.clip(raw_prob, 0.0, 1.0)

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


def _incident_entropy(incident_proba: np.ndarray) -> np.ndarray:
    """
    Reduces the incident agent's N-class softmax output to a
    single scalar per sample using Shannon entropy.

    High entropy  → the agent is uncertain across attack categories → the
                    meta-learner should treat this as an elevated risk signal.
    Low entropy   → the agent is confident about one category (or Normal) →
                    pass-through to binary decision.

    Using entropy as the reduction keeps the meta-learner's input dimensionality
    fixed regardless of how many attack classes exist in the dataset, and avoids
    overfitting that would result from feeding all N class probabilities into a
    2-weight logistic model.
    """
    # Clip to avoid log(0); incident_proba rows should already sum to 1.0
    # Ensure perfect sum to 1.0 before clipping to prevent floating-point drift
    p_norm = incident_proba / np.sum(incident_proba, axis=1, keepdims=True)
    p = np.clip(p_norm, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def _top2_margin(incident_proba: np.ndarray) -> np.ndarray:
    """
    Returns top1_prob - top2_prob for each sample.
    High margin → agent is very confident about one class (low ambiguity).
    Low margin  → agent is split between two classes (high ambiguity).
    """
    sorted_probs = np.sort(incident_proba, axis=1)[:, ::-1]  # descending
    return sorted_probs[:, 0] - sorted_probs[:, 1]

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
    # Feature order: [0] CatBoost prob, [1] RCF score, [2] Incident entropy.
    w = meta_model.weights
    print(
        f"\nMeta-Learner weights — "
        f"CatBoost: {w[0]:.4f}  RCF: {w[1]:.4f}  Entropy: {w[2]:.4f}  "
        f"Top2Margin: {w[3]:.4f}  Bias: {meta_model.bias:.4f}"
    )
    # Warn if CatBoost weight drops below 30% of total absolute weight, which
    # would indicate the precision floor is suppressing the primary classifier.
    total_abs = np.sum(np.abs(w))
    catboost_share = abs(w[0]) / total_abs if total_abs > 0 else 0.0
    if catboost_share < 0.30:
        print(
            f"[WARNING] CatBoost weight share is {catboost_share:.1%} of total — "
            f"below the 30% floor. The meta-learner may be over-relying on RCF/entropy. "
            f"Consider increasing COST_FN (currently {COST_FN}) by 2–3 to restore "
            f"recall pressure, or lowering min_precision in find_optimal_threshold."
        )
    else:
        print(f"CatBoost weight share: {catboost_share:.1%} — healthy contribution.")

    print("Meta-Learner Training Complete!")
    return meta_model