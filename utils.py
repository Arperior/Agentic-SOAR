import numpy as np
import pandas as pd
import json
import os
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import joblib


def train_categorical_model(X_cat, y, cat_cols):
    print("\nTraining CatBoost (with High Regularization)...")
    cb_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_cols,
        l2_leaf_reg=15,
        verbose=False
    )
    cb_model.fit(X_cat, y)
    return cb_model


def train_incident_agent(X_cat, y_multi, cat_cols):
    print("\nTraining Incident Agent (Multi-class Threat Classifier)...")
    incident_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        loss_function='MultiClass',
        cat_features=cat_cols,
        l2_leaf_reg=15,
        auto_class_weights='Balanced',
        verbose=False
    )
    incident_model.fit(X_cat, y_multi)
    return incident_model


def _fit_num_pipeline(X_num_raw_tr):
    """
    Fits a StandardScaler + PCA(0.95) on the provided raw numerical array/frame.
    Returns (scaler, pca, X_num_transformed_as_DataFrame).
    Kept as a helper so the exact same logic runs both inside OOF folds and
    in prepare_datasets for the final full-train artifact.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_num_raw_tr)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    pc_columns = [f"pc_{i}" for i in range(X_pca.shape[1])]
    X_num_df = pd.DataFrame(X_pca, columns=pc_columns,
                            index=X_num_raw_tr.index if hasattr(X_num_raw_tr, 'index') else None)
    return scaler, pca, X_num_df


def _transform_num_pipeline(X_num_raw, scaler, pca):
    """
    Applies a pre-fitted scaler + PCA to raw numerical data.
    Returns a DataFrame with named pc_ columns.
    """
    X_scaled = scaler.transform(X_num_raw)
    X_pca = pca.transform(X_scaled)
    pc_columns = [f"pc_{i}" for i in range(X_pca.shape[1])]
    return pd.DataFrame(X_pca, columns=pc_columns,
                        index=X_num_raw.index if hasattr(X_num_raw, 'index') else None)


def generate_oof_features(
    X_cat, X_num_raw, y, cat_cols,
    train_cat_func, rcf_class,
    train_incident_func=None, n_splits=5, y_multi=None
):
    """
    Generates clean, Out-of-Fold predictions to train the Meta-Learner.

    FIX (PCA Leakage): X_num_raw (the RAW, unscaled numerical frame) is now
    passed in instead of the PCA-transformed X_num.  A fresh StandardScaler
    and PCA are fit on X_num_tr inside every fold so the validation split
    never influences the PCA eigenvectors.  The saved scaler/PCA artifacts
    (written by prepare_datasets) are NOT used here — they are only for
    final test-set inference.

    Parameters
    ----------
    X_num_raw : pd.DataFrame
        Raw (unscaled, un-PCA'd) numerical features for the training set.
        Caller must pass X_num_raw returned by prepare_datasets, not X_num.
    """
    if train_incident_func is not None and y_multi is None:
        raise ValueError(
            "y_multi must be provided when train_incident_func is given."
        )

    print(f"\nGenerating {n_splits}-Fold OOF Predictions for Meta-Learner training...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cat = np.zeros(len(y))
    oof_rcf = np.zeros(len(y))
    oof_incident_entropy = np.zeros(len(y))

    # Reset indices for safe iloc slicing
    X_cat_reset = X_cat.reset_index(drop=True)
    X_num_raw_reset = X_num_raw.reset_index(drop=True)   # <-- raw numerics
    y_reset = y.reset_index(drop=True)
    y_multi_reset = y_multi.reset_index(drop=True) if y_multi is not None else None

    # Stratify on multiclass labels when available for better rare-class coverage
    stratify_labels = y_multi_reset if y_multi_reset is not None else y_reset

    fold = 1
    for train_idx, val_idx in skf.split(X_cat_reset, stratify_labels):
        print(f"  -> Processing Fold {fold}/{n_splits}...")

        X_cat_tr = X_cat_reset.iloc[train_idx]
        X_cat_val = X_cat_reset.iloc[val_idx]
        X_num_raw_tr = X_num_raw_reset.iloc[train_idx]
        X_num_raw_val = X_num_raw_reset.iloc[val_idx]
        y_tr = y_reset.iloc[train_idx]

        # --- PCA LEAKAGE FIX ---
        # Fit scaler + PCA exclusively on this fold's training split.
        # The validation slice is only .transform()'d — it never touches .fit().
        fold_scaler, fold_pca, X_num_tr = _fit_num_pipeline(X_num_raw_tr)
        X_num_val = _transform_num_pipeline(X_num_raw_val, fold_scaler, fold_pca)

        # 1. CatBoost binary score
        temp_cat = train_cat_func(X_cat_tr, y_tr, cat_cols)
        oof_cat[val_idx] = temp_cat.predict_proba(X_cat_val)[:, 1]

        # 2. RCF anomaly score (trained on normal traffic only)
        temp_rcf = rcf_class(num_trees=40, tree_size=256)
        X_num_tr_normal = X_num_tr[y_tr.values == 0]
        temp_rcf.fit_predict(X_num_tr_normal)
        oof_rcf[val_idx] = temp_rcf.predict_proba(X_num_val)

        # 3. Incident agent entropy (multiclass uncertainty)
        if train_incident_func is not None:
            y_multi_tr = y_multi_reset.iloc[train_idx]
            temp_incident = train_incident_func(X_cat_tr, y_multi_tr, cat_cols)
            incident_proba_val = temp_incident.predict_proba(X_cat_val)
            p = np.clip(incident_proba_val, 1e-12, 1.0)
            oof_incident_entropy[val_idx] = -np.sum(p * np.log(p), axis=1)

        fold += 1

    print("OOF Generation Complete.")
    return oof_cat, oof_rcf, oof_incident_entropy


def prepare_datasets(file_path, is_train=False):
    """
    Decoupled preprocessing to prevent data leakage.
    - is_train=True : Fits and saves Scaler/PCA on the FULL training set
                      (used only for final test-set inference, NOT for OOF).
                      Also returns X_num_raw so the caller can pass it to
                      generate_oof_features without re-reading the CSV.
    - is_train=False: Strictly loads and transforms (Zero Leakage).

    Returns
    -------
    X_cat           : pd.DataFrame  — categorical features
    X_num           : pd.DataFrame  — PCA-transformed numerical features
    X_num_raw       : pd.DataFrame  — raw (unscaled) numerical features
                      Only meaningful in train mode; in test mode it is
                      returned for API consistency but the caller should
                      use X_num directly (already transformed by saved artifacts).
    labels_binary   : pd.Series
    categorical_cols: list[str]
    labels_multiclass: pd.Series
    """
    mode_str = 'Train' if is_train else 'Test'
    print(f"Loading and transforming data from {file_path} (Mode: {mode_str})...")

    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    labels_multiclass = df['attack_cat'].fillna('Normal')
    labels_binary = df['label']
    X_raw = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')

    # --- 1. CATEGORICAL HANDLING ---
    cat_cols_standard = X_raw.select_dtypes(include=['object']).columns.tolist()
    cat_cols_deceptive = ['is_ftp_login', 'is_sm_ips_ports', 'ct_state_ttl']
    categorical_cols = sorted(
        list(set(cat_cols_standard + cat_cols_deceptive).intersection(X_raw.columns))
    )
    X_cat = X_raw[categorical_cols].copy().fillna('unknown').astype(str)

    # --- 2. NUMERICAL HANDLING ---
    numerical_cols = [col for col in X_raw.columns if col not in categorical_cols]
    X_num_raw = X_raw[numerical_cols].copy().fillna(0)

    save_dir = "Saves"
    os.makedirs(save_dir, exist_ok=True)
    scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
    pca_path = os.path.join(save_dir, "feature_pca.pkl")

    if is_train:
        # Fit on full training set and persist for test-time inference.
        # NOTE: these artifacts are intentionally NOT used during OOF generation —
        # generate_oof_features fits its own per-fold scaler/PCA from X_num_raw.
        scaler, pca, X_num = _fit_num_pipeline(X_num_raw)
        joblib.dump(scaler, scaler_path)
        joblib.dump(pca, pca_path)
    else:
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(
                f"Missing scaler/PCA artifacts in {save_dir}. Run training phase first."
            )
        scaler = joblib.load(scaler_path)
        pca = joblib.load(pca_path)
        X_num = _transform_num_pipeline(X_num_raw, scaler, pca)

    print(f"Final Categorical features: {len(categorical_cols)}")
    print(f"Numerical features processed: {X_num.shape[1]} Principal Components")

    return X_cat, X_num, X_num_raw, labels_binary, categorical_cols, labels_multiclass


def find_optimal_threshold(
    y_true,
    risk_scores,
    cost_fn: int = 10,
    cost_fp: int = 2,
    n_thresholds: int = 1000,
    min_precision: float = 0.80,
    save_path: str = "Saves/optimal_threshold.json"
) -> float:
    """
    Sweeps decision thresholds and returns the one that minimises the
    asymmetric SOC cost function:  cost = (FN × cost_fn) + (FP × cost_fp),
    subject to a minimum precision floor.

    Without a precision constraint, pure cost minimisation degenerates toward
    "flag everything" whenever attacks outnumber normals: each missed attack
    costs cost_fn while each false alarm only costs cost_fp, so setting the
    threshold near zero minimises raw cost but produces operationally useless
    results (near-100% FP rate). The min_precision floor prevents this by
    rejecting any threshold whose attack-prediction precision falls below the
    specified value.

    Parameters
    ----------
    y_true        : array-like of int   -- ground-truth binary labels (0/1)
    risk_scores   : array-like of float -- meta-learner output probabilities
    cost_fn       : int    -- penalty per missed attack (False Negative)
    cost_fp       : int    -- penalty per false alarm   (False Positive)
    n_thresholds  : int    -- number of candidate thresholds to evaluate
    min_precision : float  -- minimum acceptable precision for attack class (0-1).
                             Thresholds below this precision are skipped.
                             Default 0.80 -- at least 80% of flagged events
                             must be real attacks. Raise toward 0.90 to
                             reduce alert fatigue; lower toward 0.70 to
                             tolerate more FPs in exchange for fewer FNs.
                             NOTE: raising this value pushes the threshold
                             higher, which increases FNs. Counter this by
                             raising cost_fn so recall pressure is maintained.
    save_path     : str    -- where to persist the result (loaded by SOAR agent)

    Returns
    -------
    float -- the threshold value in [0, 1] that minimises total SOC cost
             while satisfying the precision floor
    """
    y_true = np.asarray(y_true)
    risk_scores = np.asarray(risk_scores)

    thresholds = np.linspace(0.01, 0.99, n_thresholds)
    best_threshold = 0.5
    best_cost = np.inf
    best_tn = best_fp = best_fn = best_tp = 0
    best_precision = 0.0
    n_skipped = 0

    for t in thresholds:
        preds = (risk_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

        # Precision = TP / (TP + FP); guard against zero-division when
        # no positives are predicted (very high threshold).
        predicted_positives = tp + fp
        if predicted_positives == 0:
            n_skipped += 1
            continue

        precision = tp / predicted_positives
        if precision < min_precision:
            n_skipped += 1
            continue

        cost = (fn * cost_fn) + (fp * cost_fp)
        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            best_precision = float(precision)
            best_tn, best_fp, best_fn, best_tp = tn, fp, fn, tp

    if best_cost == np.inf:
        # No threshold satisfied the precision floor -- fall back to 0.5
        # and warn loudly so the caller knows the constraint was too tight.
        print(
            f"\n[WARNING] No threshold satisfied min_precision={min_precision:.2f}. "
            f"All {n_thresholds} candidates were skipped. "
            f"Falling back to 0.5 -- consider lowering min_precision."
        )
        best_threshold = 0.5
        preds = (risk_scores >= 0.5).astype(int)
        best_tn, best_fp, best_fn, best_tp = confusion_matrix(y_true, preds).ravel()
        best_cost = int((best_fn * cost_fn) + (best_fp * cost_fp))
        best_precision = float(best_tp / max(best_tp + best_fp, 1))

    print("\n--- THRESHOLD OPTIMISATION (OOF) ---")
    print(f"Sweep range    : 0.01 - 0.99 ({n_thresholds} candidates, {n_skipped} below precision floor)")
    print(f"Cost function  : (FN x {cost_fn}) + (FP x {cost_fp})")
    print(f"Precision floor: {min_precision:.2f} (min fraction of flagged events that must be real attacks)")
    print(f"Optimal threshold    : {best_threshold:.4f}")
    print(f"Precision @ threshold: {best_precision:.4f}")
    print(f"Minimum SOC cost     : {best_cost}")
    print(f"  TN={best_tn}  FP={best_fp}  FN={best_fn}  TP={best_tp}")

    # Persist so the SOAR agent can load it without re-running training
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    record = {
        "optimal_threshold": float(best_threshold),
        "soc_cost":          int(best_cost),
        "cost_fn":           int(cost_fn),
        "cost_fp":           int(cost_fp),
        "min_precision":     float(min_precision),
        "achieved_precision": float(best_precision),
        "confusion": {
            "tn": int(best_tn), "fp": int(best_fp),
            "fn": int(best_fn), "tp": int(best_tp)
        }
    }
    with open(save_path, "w") as f:
        json.dump(record, f, indent=4)
    print(f"Threshold saved to {save_path}")

    return best_threshold


def validate_results(y_true, final_risk, catboost_model, categorical_cols,
                     optimal_threshold: float = 0.5):
    print("\n--- PIPELINE VALIDATION ---")

    predictions = (final_risk > optimal_threshold).astype(int)

    print("\n1. Classification Report (Operational Impact):")
    print(classification_report(y_true, predictions, target_names=['Normal (0)', 'Attack (1)']))

    cm = confusion_matrix(y_true, predictions)
    print("\n2. Confusion Matrix:")
    print(f"True Negatives (Correctly Allowed):  {cm[0][0]}")
    print(f"False Positives (Unjustified Blocks): {cm[0][1]}  <-- Alert Fatigue Risk")
    print(f"False Negatives (Missed Attacks):     {cm[1][0]}  <-- Security Breach Risk")
    print(f"True Positives (Correctly Blocked):   {cm[1][1]}")

    print("\n3. Top 5 Drivers of Categorical Risk:")
    feature_importances = catboost_model.get_feature_importance()
    importance_dict = dict(zip(categorical_cols, feature_importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    for feature, importance in sorted_importance[:5]:
        print(f" - {feature}: {importance:.2f}% influence")