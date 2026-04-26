import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
import joblib
import os

def train_categorical_model(X_cat, y, cat_cols):
    print("\nTraining CatBoost (with High Regularization)...")
    cb_model = CatBoostClassifier(
        iterations=100,
        learning_rate=0.1,
        depth=6,
        cat_features=cat_cols,
        l2_leaf_reg=15,  #Forces the model to diversify features
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
        auto_class_weights='Balanced',  # prevents rare attacks from being ignored
        verbose=False 
    )
    incident_model.fit(X_cat, y_multi)
    return incident_model

def generate_oof_features(X_cat, X_num, y, cat_cols, train_cat_func, rcf_class, train_incident_func, n_splits=5, y_multi=None):
    """
    Generates clean, Out-of-Fold predictions to train the Meta-Learner without leakage.

    FIX (Issue 3): Also generates OOF incident-agent entropy so the meta-learner
    trains on the same 3-feature distribution it receives at test time.

    FIX (Issue 5): All three OOF arrays are guaranteed to have the same length
    as `y`, so np.column_stack in the caller will always produce a valid (N, 3)
    matrix — a shape mismatch here would previously surface only at runtime as a
    cryptic broadcast error.

    Parameters
    ----------
    y_multi : pd.Series, required when train_incident_func is provided.
        The multiclass attack-category labels aligned with y. Must be passed in
        so the incident agent folds are sliced correctly — using y_tr (binary)
        here would cause CatBoostClassifier to train with loss_function='MultiClass'
        on binary labels, which raises a silent accuracy collapse or a hard error.
    """
    if train_incident_func is not None and y_multi is None:
        raise ValueError(
            "y_multi must be provided when train_incident_func is given. "
            "Pass the multiclass label Series from prepare_datasets()."
        )

    print(f"\nGenerating {n_splits}-Fold OOF Predictions for Meta-Learner training...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    oof_cat = np.zeros(len(y))
    oof_rcf = np.zeros(len(y))
    oof_incident_entropy = np.zeros(len(y))

    # Reset indices to ensure smooth alignment during slicing
    X_cat_reset = X_cat.reset_index(drop=True)
    X_num_reset = X_num.reset_index(drop=True)
    y_reset = y.reset_index(drop=True)
    y_multi_reset = y_multi.reset_index(drop=True) if y_multi is not None else None

    fold = 1
    for train_idx, val_idx in skf.split(X_cat_reset, y_reset):
        print(f"  -> Processing Fold {fold}/{n_splits}...")
        
        X_cat_tr, X_cat_val = X_cat_reset.iloc[train_idx], X_cat_reset.iloc[val_idx]
        X_num_tr, X_num_val = X_num_reset.iloc[train_idx], X_num_reset.iloc[val_idx]
        y_tr = y_reset.iloc[train_idx]

        # 1. Train temporary CatBoost and predict on hold-out fold
        temp_cat = train_cat_func(X_cat_tr, y_tr, cat_cols)
        oof_cat[val_idx] = temp_cat.predict_proba(X_cat_val)[:, 1]

        # 2. Train temporary RCF (ONLY on normal traffic) and predict blindly on hold-out
        temp_rcf = rcf_class(num_trees=40, tree_size=256)
        X_num_tr_normal = X_num_tr[y_tr == 0]
        temp_rcf.fit_predict(X_num_tr_normal) 
        oof_rcf[val_idx] = temp_rcf.predict_proba(X_num_val)

        # 3. FIX (Issue 3): Train temporary incident agent on the MULTICLASS labels
        # (y_multi_tr), not the binary y_tr — passing binary labels to a MultiClass
        # CatBoost model either silently collapses to a 2-class problem or raises
        # a dimension error depending on the CatBoost version.
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
    - is_train=True: Fits and saves Scaler/PCA.
    - is_train=False: Strictly loads and transforms (Zero Leakage).
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
    
    # CRITICAL FIX: Sorted prevents non-deterministic column ordering which breaks CatBoost
    categorical_cols = sorted(list(set(cat_cols_standard + cat_cols_deceptive).intersection(X_raw.columns)))
    
    X_cat = X_raw[categorical_cols].copy()
    X_cat = X_cat.fillna('unknown').astype(str)

    # --- 2. NUMERICAL HANDLING ---
    numerical_cols = [col for col in X_raw.columns if col not in categorical_cols]
    X_num_raw = X_raw[numerical_cols].copy().fillna(0)
    
    # Ensure the Saves directory exists to prevent FileNotFoundError
    save_dir = "Saves"
    os.makedirs(save_dir, exist_ok=True)
    
    scaler_path = os.path.join(save_dir, "feature_scaler.pkl")
    pca_path = os.path.join(save_dir, "feature_pca.pkl")

    if is_train:
        # Fit and Save for future use (Training Phase)
        scaler = StandardScaler()
        X_num_scaled = scaler.fit_transform(X_num_raw)
        joblib.dump(scaler, scaler_path)

        pca = PCA(n_components=0.95, random_state=42)
        X_num_pca = pca.fit_transform(X_num_scaled)
        joblib.dump(pca, pca_path)
    else:
        # Load and Transform ONLY (Test/Inference Phase)
        if not os.path.exists(scaler_path) or not os.path.exists(pca_path):
            raise FileNotFoundError(f"Missing files in {save_dir}. Run training phase first.")
            
        scaler = joblib.load(scaler_path)
        X_num_scaled = scaler.transform(X_num_raw)

        pca = joblib.load(pca_path)
        X_num_pca = pca.transform(X_num_scaled)
    
    # CRITICAL FIX: Preserve feature alignment by naming the PCA columns explicitly
    pc_columns = [f"pc_{i}" for i in range(X_num_pca.shape[1])]
    X_num = pd.DataFrame(X_num_pca, columns=pc_columns)

    print(f"Final Categorical features: {len(categorical_cols)}")
    print(f"Numerical features processed: {X_num.shape[1]} Principal Components")
    
    return X_cat, X_num, labels_binary, categorical_cols, labels_multiclass

def validate_results(y_true, final_risk, catboost_model, categorical_cols):
    print("\n--- PIPELINE VALIDATION ---")
    
    predictions = (final_risk > 0.5).astype(int)
    
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
    
    # Pair features with their importance scores and sort them
    importance_dict = dict(zip(categorical_cols, feature_importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_importance[:5]:
        print(f" - {feature}: {importance:.2f}% influence")