import rrcf
import numpy as np
import tqdm
import dill as pickle
import os
from collections import deque
import uuid

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
        # Percentile anchors for score normalization — set at end of fit_predict.
        self._score_p1 = None
        self._score_p99 = None

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
        # FIX: Generate a unique ID for this specific thread's evaluation
        temp_index = f"temp_eval_{uuid.uuid4().hex}" 
        
        for tree in self.forest:
            tree.insert_point(x, index=temp_index)
            scores.append(tree.codisp(temp_index))
            tree.forget_point(temp_index) 
            
        return float(np.mean(scores))

    def _normalize_score(self, raw_score):
        """
        Maps a raw codisp score to [0, 1].

        The fixed sigmoid (1 / (1 + exp(-log1p(raw)))) saturates near 0.98
        for any codisp value above ~50, which is common in UNSW-NB15.
        This makes normal and attack traffic indistinguishable at the output
        layer and collapses the meta-learner threshold toward zero.

        When _score_p1 and _score_p99 are available (set after fit_predict
        finishes), we use a linear percentile stretch instead:
            normalized = clip((raw - p1) / (p99 - p1), 0, 1)
        This maps the 1st-percentile score to ~0.0 and the 99th to ~1.0,
        preserving the full dynamic range without saturation.

        FIX (Issue 4 — RCF ceiling saturation): The RCF is trained on normal
        traffic only.  Attack traffic at test time often produces codisp values
        far above the training p99, so the clip at 1.0 causes many attack rows
        to saturate at exactly 1.0000 — collapsing the score distribution and
        making them indistinguishable from each other.  To give the meta-learner
        more signal above the 99th percentile, the ceiling is relaxed to
        _score_p99 * 1.5 (a 50% headroom above the highest normal-traffic score
        seen during training).  This spreads the high-anomaly region from a
        single cliff edge at 1.0 into a graded ramp, while still capping the
        very highest outliers to prevent unbounded scores from destabilising the
        meta-learner's logistic function.
        Falls back to the sigmoid only during fit_predict before calibration.
        """
        if self._score_p1 is not None and self._score_p99 is not None:
            # FIX: use p99 * 1.5 as ceiling to reduce attack-score saturation.
            # p1 stays as floor (maps the lowest normal scores to ~0.0).
            effective_ceiling = self._score_p99 * 1.5
            span = effective_ceiling - self._score_p1
            if span > 0:
                return float(np.clip((raw_score - self._score_p1) / span, 0.0, 1.0))
        # Pre-calibration fallback (only hit during fit_predict before stats are set)
        return float(1.0 / (1.0 + np.exp(-np.log1p(raw_score))))

    def _smooth_score(self, score):
        # FIX (Issue 2): deque(maxlen=N) automatically evicts oldest entry —
        # no manual pop needed, and the window is strictly bounded.
        self.history.append(score)
        return float(np.mean(self.history))

    # --- CORE API METHODS ---

    def fit_predict(self, X,global_p1=None, global_p99=None):
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
        raw_scores = []
        X_array = np.asarray(X)
        
        for x in tqdm.tqdm(X_array, desc="RRCF Training & Scoring"):
            self.index += 1
            raw_score = self._insert_point_and_score(x)
            raw_scores.append(raw_score)

        # Calibrate percentile anchors on the training distribution before
        # normalizing. This stretches the score range so normal traffic maps
        # near 0 and genuine anomalies map near 1, preventing sigmoid saturation.
        # FIX (Issue 5 — OOF/test cost gap): Print the anchors so the caller
        # can compare per-fold anchors (from generate_oof_features) against the
        # final full-train RCF anchors.  A large divergence means OOF meta-learner
        # features were generated on a different score scale than what the final
        # RCF produces at test time, inflating the train→test cost gap.
        raw_array = np.array(raw_scores)
        # FIX: If global anchors are provided (from the full-train model), use them!
        # Otherwise, calculate them locally.
        if global_p1 is not None and global_p99 is not None:
            self._score_p1 = global_p1
            self._score_p99 = global_p99
        else:
            self._score_p1  = float(np.percentile(raw_array, 1))
            self._score_p99 = float(np.percentile(raw_array, 99))

        print(
            f"[RCF] Calibration anchors — "
            f"p1={self._score_p1:.4f}  p99={self._score_p99:.4f}  "
            f"effective_ceiling={self._score_p99 * 1.5:.4f}  "
            f"n_samples={len(raw_array)}"
        )

        self._is_fitted = True
        return np.array([self._normalize_score(r) for r in raw_scores])

    def predict_proba(self, X, smooth=False):
        """
        Phase 4 (Batch Testing): Scores unseen data blindly.
        smooth=False guarantees order-independent, i.i.d evaluation and matches
        the distribution used during OOF generation in fit_predict().

        FIX (Issue 4): Prints a saturation diagnostic after scoring so the caller
        can verify that attack scores are not collapsing to 1.0000 en masse.
        If "% at ceiling" is above ~15%, the effective_ceiling in _normalize_score
        should be increased (e.g. p99 * 2.0) or the RCF retrained.
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

        score_array = np.array(scores)

        # --- Saturation diagnostic ---
        pct_ceiling = (score_array >= 0.999).mean() * 100
        print(
            f"\n[RCF] Score distribution — "
            f"min={score_array.min():.4f}  mean={score_array.mean():.4f}  "
            f"max={score_array.max():.4f}  "
            f"% at ceiling (≥0.999): {pct_ceiling:.1f}%"
        )
        if pct_ceiling > 15.0:
            print(
                f"[RCF] WARNING: {pct_ceiling:.1f}% of scores saturated at ceiling. "
                f"Consider increasing effective_ceiling multiplier in _normalize_score "
                f"(currently p99 × 1.5) or retraining with a wider score range."
            )

        return score_array

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
        self._score_p1 = None
        self._score_p99 = None

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
            'history': list(self.history),
            # Percentile anchors must travel with the model so predict_proba
            # uses the same normalization scale as fit_predict.
            'score_p1':  self._score_p1,
            'score_p99': self._score_p99,
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
        instance.history = deque(state.get('history', []), maxlen=state['smoothing_window'])
        # Restore percentile anchors so normalization is consistent with training.
        # .get() fallback handles pkl files saved before this fix (uses sigmoid).
        instance._score_p1  = state.get('score_p1')
        instance._score_p99 = state.get('score_p99')
        instance._is_fitted = True
        
        print(f"RCF state successfully loaded from {filepath}")
        return instance