import rrcf
import numpy as np
import tqdm

def score_rcf_streaming(X_num):
    rcf_model = RCF(num_trees=40, tree_size=256, warmup=500, smoothing_window=20)
    rcf_scores = []
    
    for row in tqdm.tqdm(X_num.values, desc="RRCF Scoring Progress"): 
        score = rcf_model.score(row)
        rcf_scores.append(score)
        
    rcf_scores = np.array(rcf_scores)
    return rcf_scores

class RCF:
    def __init__(self, num_trees=40, tree_size=256, seed=0, warmup=500, smoothing_window=20):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.rng = np.random.default_rng(seed)

        self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        self.index = 0

        # Warm-up period before scores are considered reliable
        self.warmup = warmup

        # For smoothing anomaly scores
        self.smoothing_window = smoothing_window
        self.history = []

    def _insert_point(self, x):
        scores = []

        for tree in self.forest:
            # Maintain tree size
            if len(tree.leaves) >= self.tree_size:
                drop_index = self.rng.choice(list(tree.leaves.keys()))
                tree.forget_point(drop_index)

            tree.insert_point(x, index=self.index)
            scores.append(tree.codisp(self.index))

        return float(np.mean(scores))

    def _normalize_score(self, raw_score):
        # Stabilize using log + sigmoid transformation
        return 1.0 / (1.0 + np.exp(-np.log1p(raw_score)))

    def _smooth_score(self, score):
        self.history.append(score)

        if len(self.history) > self.smoothing_window:
            self.history.pop(0)

        return float(np.mean(self.history))

    def fit(self, X):
        """
        Builds the forest using initial data (warm-up phase).
        No meaningful anomaly scores are returned here.
        """
        for x in X:
            self.index += 1
            self._insert_point(x)

    def score(self, x):
        """
        Insert a new point and return a smoothed anomaly score.
        """
        self.index += 1

        raw_score = self._insert_point(x)
        norm_score = self._normalize_score(raw_score)

        # During warm-up phase, suppress unreliable scores
        if self.index < self.warmup:
            return 0.0

        smooth_score = self._smooth_score(norm_score)
        return smooth_score

    def score_batch(self, X):
        """
        Score multiple samples sequentially.
        """
        scores = []
        for x in X:
            scores.append(self.score(x))
        return np.array(scores)

    def reset(self):
        """
        Reset the model to initial state.
        """
        self.forest = [rrcf.RCTree() for _ in range(self.num_trees)]
        self.index = 0
        self.history = []