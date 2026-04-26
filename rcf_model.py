import rrcf
import numpy as np
import tqdm
import dill as pickle
import os
from collections import deque

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
        # FIX (Issue 2): Use a fixed-length deque so a sustained burst of anomalies
        # cannot erode detection sensitivity by growing the history buffer unboundedly.
        self.history = deque(maxlen=smoothing_window)
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
        # FIX (Issue 2): deque(maxlen=N) automatically evicts oldest entry —
        # no manual pop needed, and the window is strictly bounded.
        self.history.append(score)
        return float(np.mean(self.history))

    # --- CORE API METHODS ---

    def fit_predict(self, X):
        """
        Phase 1 (Training): Builds the baseline forest from normal traffic.

        FIX (Issue 1 — Score Distribution Mismatch):
        Returns raw, unsmoothed scores so OOF features fed to the meta-learner
        are generated under IDENTICAL conditions to predict_proba() at inference
        time. Previously, fit_predict used smoothing and warmup suppression while
        predict_proba used smooth=False, causing the meta-learner to be trained on
        a different score distribution than it receives during evaluation.

        Warmup suppression (returning 0.0 for the first N points) is also removed:
        injecting artificial zeros into OOF features would skew the meta-learner's
        weight calibration. The forest's early scores are naturally low-confidence
        but are real values the meta-learner can learn from.
        """
        print(f"Building RRCF Forest ({self.num_trees} trees)...")
        scores = []
        X_array = np.asarray(X)
        
        for x in tqdm.tqdm(X_array, desc="RRCF Training & Scoring"):
            self.index += 1
            raw_score = self._insert_point_and_score(x)
            norm_score = self._normalize_score(raw_score)
            scores.append(norm_score)  # Always append the real score
                
        self._is_fitted = True
        return np.array(scores)

    def predict_proba(self, X, smooth=False):
        """
        Phase 4 (Batch Testing): Scores unseen data blindly.
        smooth=False guarantees order-independent, i.i.d evaluation and matches
        the distribution used during OOF generation in fit_predict().
        """
        if not self._is_fitted:
            raise RuntimeError("[ERROR] RCF Model must be fitted before predicting.")
            
        scores = []
        X_array = np.asarray(X)
        
        for x in tqdm.tqdm(X_array, desc="RRCF Blind Test Scoring"):
            raw_score = self._score_blind(x)
            norm_score = self._normalize_score(raw_score)
            
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
        self.history = deque(maxlen=self.smoothing_window)
        self._is_fitted = False

    # --- PERSISTENCE METHODS ---
    def save_model(self, filepath=r"Saves/rcf_base.pkl"):
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")
            
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
        state = {
            'num_trees': self.num_trees,
            'tree_size': self.tree_size,
            'warmup': self.warmup,
            'smoothing_window': self.smoothing_window,
            'forest': self.forest,
            'index': self.index,
            # FIX (Issue 4): Persist history so the loaded model resumes with a
            # warm smoothing context rather than a cold-start deque. Without this,
            # the first `smoothing_window` live predictions after loading have a
            # systematic cold-start bias toward lower (less anomalous) scores.
            'history': list(self.history),
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
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
        # FIX (Issue 4): Restore history as a bounded deque, not a plain list,
        # so the smoothing window behaves identically to the trained instance.
        # .get() with a fallback handles pkl files saved before this fix.
        instance.history = deque(state.get('history', []), maxlen=state['smoothing_window'])
        instance._is_fitted = True
        
        print(f"RCF state successfully loaded from {filepath}")
        return instance