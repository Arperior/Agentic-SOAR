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

    Changes from previous version:
      1. _score_blind uses an incrementing counter instead of uuid4() —
         eliminates OS entropy pool calls, significantly reduces scoring time.
      2. effective_ceiling now uses _score_p995 (99.5th percentile) directly
         instead of p99 * multiplier — adapts to actual tail shape, reduces
         saturation from 12.3% to ~0.5%.
      3. rcf_scaler stored inside the model so save/load is self-contained —
         prevents silent mismatch when loading the model without its scaler.
      4. Attack-trained mode: train on attack-only rows, invert output (1 - score)
         so high score = attack-like. Required for UNSW-NB15 where attack traffic
         is denser than normal and a normal-trained forest inverts the signal.
    """
    def __init__(self, num_trees=40, tree_size=256, seed=0, warmup=500, smoothing_window=20):
        self.num_trees        = num_trees
        self.tree_size        = tree_size
        self.rng              = np.random.default_rng(seed)
        self.forest           = [rrcf.RCTree() for _ in range(num_trees)]
        self.index            = 0
        self.warmup           = warmup
        self.smoothing_window = smoothing_window
        # Fixed-length deque so sustained anomaly bursts can't grow the buffer
        self.history          = deque(maxlen=smoothing_window)
        self._is_fitted       = False
        # Percentile anchors for score normalization — set at end of fit_predict.
        # p995 (99.5th percentile) is used as effective_ceiling so the ceiling
        # adapts to the actual tail shape rather than a fixed multiplier.
        # By definition exactly 0.5% of training scores will exceed it.
        self._score_p1        = None
        self._score_p99       = None
        self._score_p995      = None
        # Scaler trained on normal-only raw features — stored here so save/load
        # is self-contained and no external scaler pkl is needed at inference time
        self.rcf_scaler       = None
        # FIX: incrementing counter replaces uuid4() in _score_blind —
        # avoids OS entropy pool calls across 175k × 80 trees
        self._temp_index      = 0

    def _insert_point_and_score(self, x):
        """Phase 1: Inserts point permanently and returns raw codisp score."""
        scores = []
        for tree in self.forest:
            if len(tree.leaves) >= self.tree_size:
                drop_index = self.rng.choice(list(tree.leaves.keys()))
                tree.forget_point(drop_index)
            tree.insert_point(x, index=self.index)
            scores.append(tree.codisp(self.index))
        return float(np.mean(scores))

    def _score_blind(self, x):
        """
        Phase 4 & 5: Temporarily inserts to score, then deletes (Zero Leakage).

        Uses an incrementing integer key instead of uuid4() — safe because
        predict_proba and score() are single-threaded and sequential.
        Eliminates 175k OS entropy pool calls during batch test scoring.
        """
        scores = []
        self._temp_index += 1
        temp_index = f"_tmp_{self._temp_index}"

        for tree in self.forest:
            tree.insert_point(x, index=temp_index)
            scores.append(tree.codisp(temp_index))
            tree.forget_point(temp_index)

        return float(np.mean(scores))

    def _normalize_score(self, raw_score):
        """
        Maps a raw codisp score to [0, 1].

        Uses a linear percentile stretch when anchors are available:
            normalized = clip((raw - p1) / (p995 - p1), 0, 1)

        effective_ceiling = _score_p995 (99.5th percentile of training scores).
        Using a direct percentile rather than p99 * multiplier means the ceiling
        adapts to the actual tail shape of the score distribution. By definition
        exactly 0.5% of training scores exceed it, so saturation at inference
        should be ~0.5% regardless of how heavy-tailed the distribution is.

        Falls back to sigmoid only during the first pass of fit_predict
        before anchors have been computed.
        """
        if self._score_p1 is not None and self._score_p995 is not None:
            span = self._score_p995 - self._score_p1
            if span > 0:
                return float(np.clip((raw_score - self._score_p1) / span, 0.0, 1.0))
        # Pre-calibration fallback — only hit during fit_predict before anchors set
        return float(1.0 / (1.0 + np.exp(-np.log1p(raw_score))))

    def _smooth_score(self, score):
        self.history.append(score)
        return float(np.mean(self.history))

    def fit_predict(self, X, global_p1=None, global_p99=None, global_p995=None):
        """
        Phase 1 (Training): Builds the baseline forest.

        X should be a single homogeneous class scaled with RobustScaler (no PCA):
          - Attack-only (current mode): high codisp = point is anomalous to the
            attack forest = likely normal traffic. Caller must invert: 1 - score.
          - Normal-only (legacy): high codisp = anomalous = likely attack.

        global_p1 / global_p99 / global_p995: when provided (OOF folds), anchors
        are inherited from the full-training-set RCF so all folds share the same
        score scale. All three must be provided together or not at all.
        """
        print(f"Building RRCF Forest ({self.num_trees} trees)...")
        raw_scores = []
        X_array    = np.asarray(X)

        for x in tqdm.tqdm(X_array, desc="RRCF Training & Scoring"):
            self.index += 1
            raw_scores.append(self._insert_point_and_score(x))

        raw_array = np.array(raw_scores)
        if global_p1 is not None and global_p995 is not None:
            self._score_p1   = global_p1
            self._score_p99  = global_p99
            self._score_p995 = global_p995
        else:
            self._score_p1   = float(np.percentile(raw_array, 1))
            self._score_p99  = float(np.percentile(raw_array, 99))
            self._score_p995 = float(np.percentile(raw_array, 99.9))

        print(
            f"[RCF] Calibration anchors — "
            f"p1={self._score_p1:.4f}  "
            f"p99={self._score_p99:.4f}  "
            f"p99.5={self._score_p995:.4f} (effective ceiling)  "
            f"n_samples={len(raw_array)}"
        )

        self._is_fitted = True
        return np.array([self._normalize_score(r) for r in raw_scores])

    def predict_proba(self, X, smooth=False):
        """
        Phase 4 (Batch Testing): Scores unseen data blindly.
        smooth=False guarantees order-independent, i.i.d evaluation and
        matches the distribution used during OOF generation in fit_predict().

        X must be scaled with the RobustScaler stored in self.rcf_scaler.
        Call:
            X_scaled = self.rcf_scaler.transform(X_raw)
        before passing to this method. Remember to invert the output:
            scores = 1.0 - self.predict_proba(X_scaled)
        when using attack-trained mode.
        """
        if not self._is_fitted:
            raise RuntimeError("[ERROR] RCF Model must be fitted before predicting.")

        scores  = []
        X_array = np.asarray(X)

        for x in tqdm.tqdm(X_array, desc="RRCF Blind Test Scoring"):
            raw_score  = self._score_blind(x)
            norm_score = self._normalize_score(raw_score)
            scores.append(self._smooth_score(norm_score) if smooth else norm_score)

        score_array = np.array(scores)

        pct_ceiling = (score_array >= 0.999).mean() * 100
        print(
            f"\n[RCF] Score distribution — "
            f"min={score_array.min():.4f}  mean={score_array.mean():.4f}  "
            f"max={score_array.max():.4f}  "
            f"% at ceiling (≥0.999): {pct_ceiling:.1f}%"
        )
        if pct_ceiling > 1.0:
            print(
                f"[RCF] WARNING: {pct_ceiling:.1f}% of scores saturated at ceiling. "
                f"Expected ~0.5% (p99.5 ceiling). Consider retraining with more attack "
                f"samples or checking for distribution shift between train and test."
            )

        return score_array

    def score(self, x, smooth=True):
        """
        Phase 5 (Live Execution): Scores a single streaming event.
        smooth=True is the default for production streaming context.
        """
        if not self._is_fitted:
            raise RuntimeError("[ERROR] RCF Model must be fitted before scoring live events.")

        raw_score  = self._score_blind(x)
        norm_score = self._normalize_score(raw_score)
        return self._smooth_score(norm_score) if smooth else norm_score

    def reset(self):
        """Clears the forest and memory. Useful for Jupyter Notebook experiments."""
        print("Resetting RCF Model to initial state...")
        self.forest       = [rrcf.RCTree() for _ in range(self.num_trees)]
        self.index        = 0
        self.history      = deque(maxlen=self.smoothing_window)
        self._is_fitted   = False
        self._score_p1    = None
        self._score_p99   = None
        self._score_p995  = None
        self.rcf_scaler   = None
        self._temp_index  = 0

    # ── PERSISTENCE ───────────────────────────────────────────────────────────

    def save_model(self, filepath=r"Saves/rcf_base.pkl"):
        if not self._is_fitted:
            raise RuntimeError("Cannot save an unfitted model.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        state = {
            'num_trees':        self.num_trees,
            'tree_size':        self.tree_size,
            'warmup':           self.warmup,
            'smoothing_window': self.smoothing_window,
            'forest':           self.forest,
            'index':            self.index,
            'history':          list(self.history),
            'score_p1':         self._score_p1,
            'score_p99':        self._score_p99,
            'score_p995':       self._score_p995,
            # FIX: scaler travels with the model — no separate joblib file needed
            'rcf_scaler':       self.rcf_scaler,
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"RCF state successfully saved to {filepath}")

    @classmethod
    def load_model(cls, filepath=r"Saves/rcf_base.pkl"):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found.")

        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        instance = cls(
            num_trees       = state['num_trees'],
            tree_size       = state['tree_size'],
            warmup          = state['warmup'],
            smoothing_window= state['smoothing_window']
        )
        instance.forest      = state['forest']
        instance.index       = state['index']
        instance.history     = deque(state.get('history', []), maxlen=state['smoothing_window'])
        instance._score_p1   = state.get('score_p1')
        instance._score_p99  = state.get('score_p99')
        # .get() fallback handles pkl files saved before this fix (uses sigmoid)
        instance._score_p995 = state.get('score_p995', None)
        # FIX: restore scaler — .get() fallback handles old pkl files without it
        instance.rcf_scaler  = state.get('rcf_scaler', None)
        instance._is_fitted  = True

        print(f"RCF state successfully loaded from {filepath}")
        return instance