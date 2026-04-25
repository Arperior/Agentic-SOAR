import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

def prepare_datasets(file_path):
    print(f"Loading and transforming data from {file_path}...")
    df = pd.read_csv(file_path)

    if 'id' in df.columns:
        df = df.drop('id', axis=1)

    labels_multiclass = df['attack_cat'].fillna('Normal')
    labels_binary = df['label']
    X_raw = df.drop(['label', 'attack_cat'], axis=1, errors='ignore')

    # CATEGORICAL 
    cat_cols_standard = X_raw.select_dtypes(include=['object']).columns.tolist()
    cat_cols_deceptive = ['is_ftp_login', 'is_sm_ips_ports', 'ct_state_ttl'] 
    
    categorical_cols = list(set(cat_cols_standard + cat_cols_deceptive).intersection(X_raw.columns))
    
    X_train_cat = X_raw[categorical_cols].copy()
    X_train_cat = X_train_cat.fillna('unknown').astype(str)

    # NUMERICAL
    numerical_cols = [col for col in X_raw.columns if col not in categorical_cols]
    X_train_num_raw = X_raw[numerical_cols].copy().fillna(0)
    
    scaler = StandardScaler()
    X_num_scaled = scaler.fit_transform(X_train_num_raw)
    pca = PCA(n_components=0.95, random_state=42)
    X_train_num_pca = pca.fit_transform(X_num_scaled)
    
    X_train_num = pd.DataFrame(X_train_num_pca)

    print(f"Final Categorical features: {len(categorical_cols)}")
    print(f"Numerical features compressed from {len(numerical_cols)} down to {X_train_num.shape[1]} Principal Components")
    
    return X_train_cat, X_train_num, labels_binary, categorical_cols, labels_multiclass

def validate_results(y_true, final_risk, catboost_model, categorical_cols,optimal_threshold):
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
    
    # Pair features with their importance scores and sort them
    importance_dict = dict(zip(categorical_cols, feature_importances))
    sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    for feature, importance in sorted_importance[:5]:
        print(f" - {feature}: {importance:.2f}% influence")