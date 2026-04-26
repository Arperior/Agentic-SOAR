import rrcf
import numpy as np
import tqdm
import dill as pickle
import os

class RCF:
    """
    Robust Random Cut Forest with Streaming Warm-up and Smoothing.
    With strict Train/Test isolation, Live Streaming, and Disk Persistence.
    """
    def __init__(self, num_trees=40, tree_size=256, seed=0, warmup=500, smoothing_window=20):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.rng = np.random.default_rng(seed)
        
        self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        self.index = 0
        self.warmup = warmup
        self.smoothing_window = smoothing_window
        self.history = []
        self._is_fitted = False

    def _insert_point_and_score(self, x):
        """Phase 1: Inserts point permanently and returns raw score."""
        scores = []
        for tree in self.forest:
            if len(tree.leaves) >= self.tree_size:
                drop_index = self.rng.choice(list(tree.leaves.keys()))
                tree.forget_point(drop_index)

            tree.insert_point(x, index=self.index)
            scores.append(tree.codisp(self.index))
            
        return float(np.mean(scores))

    def _score_blind(self, x):
        """Phase 4 & 5: Temporarily inserts to score, then deletes (Zero Leakage)."""
        scores = []
        temp_index = "temp_eval_node"
        
        for tree in self.forest:
            tree.insert_point(x, index=temp_index)
            scores.append(tree.codisp(temp_index))
            tree.forget_point(temp_index) 
            
        return float(np.mean(scores))

    def _normalize_score(self, raw_score):
        return 1.0 / (1.0 + np.exp(-np.log1p(raw_score)))

    def _smooth_score(self, score):
        self.history.append(score)
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)
        return float(np.mean(self.history))

    # --- CORE API METHODS ---

    def fit_predict(self, X):
        """Phase 1 (Training): Builds baseline forest and returns scores."""
        print(f"Building RRCF Forest ({self.num_trees} trees)...")
        scores = []
        X_array = np.asarray(X)
        
        for x in tqdm.tqdm(X_array, desc="RRCF Training & Scoring"):
            self.index += 1
            raw_score = self._insert_point_and_score(x)
            norm_score = self._normalize_score(raw_score)

            if self.index < self.warmup:
                scores.append(0.0)
            else:
                scores.append(self._smooth_score(norm_score))
                
        self._is_fitted = True
        return np.array(scores)

    def predict_proba(self, X, smooth=False):
        """
        Phase 4 (Batch Testing): Scores unseen data blindly.
        smooth=False guarantees order-independent, i.i.d evaluation.
        """
        if not self._is_fitted:
            raise RuntimeError("[ERROR] RCF Model must be fitted before predicting.")
            
        self.history = [] 
        scores = []
        X_array = np.asarray(X)
        
        for x in tqdm.tqdm(X_array, desc="RRCF Blind Test Scoring"):
            raw_score = self._score_blind(x)
            norm_score = self._normalize_score(raw_score)
            
            # Apply your context-aware smoothing logic
            if smooth:
                scores.append(self._smooth_score(norm_score))
            else:
                scores.append(norm_score)
            
        return np.array(scores)

    def score(self, x, smooth=True):
        """
        Phase 5 (Live Execution): Scores a single streaming event.
        smooth=True is the default for production streaming context.
        """
        if not self._is_fitted:
            raise RuntimeError("[ERROR] RCF Model must be fitted before scoring live events.")
            
        raw_score = self._score_blind(x)
        norm_score = self._normalize_score(raw_score)
        
        if smooth:
            return self._smooth_score(norm_score)
        return norm_score

    def reset(self):
        """Clears the forest and memory. Useful for Jupyter Notebook experiments."""
        print("Resetting RCF Model to initial state...")
        self.forest = [rrcf.RCTree() for _ in range(self.num_trees)]
        self.index = 0
        self.history = []
        self._is_fitted = False

    # --- PERSISTENCE METHODS ---
    def save_model(self, filepath=r"Saves/rcf_base.pkl"):
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
            
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        state = {
            'num_trees': self.num_trees,
            'tree_size': self.tree_size,
            'warmup': self.warmup,
            'smoothing_window': self.smoothing_window,
            'forest': self.forest,
            'index': self.index
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)  # Using dill's dump
        print(f"RCF state successfully saved to {filepath}")

    @classmethod
    def load_model(cls, filepath=r"Saves/rcf_base.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")
            
        with open(filepath, 'rb') as f:
            state = pickle.load(f)  # Using dill's load
            
        instance = cls(
            num_trees=state['num_trees'], 
            tree_size=state['tree_size'],
            warmup=state['warmup'],
            smoothing_window=state['smoothing_window']
        )
        instance.forest = state['forest']
        instance.index = state['index']
        instance._is_fitted = True
        
        print(f"RCF state successfully loaded from {filepath}")
        return instance