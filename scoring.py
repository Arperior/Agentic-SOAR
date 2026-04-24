import numpy as np
from sklearn.metrics import confusion_matrix


def risk_fusion_scorer(cat_risk_scores, rcf_risk_scores, cat_weight=0.5):
    """
    Fuses the scores using a dynamic hyperparameter.
    cat_weight: float between 0.0 and 1.0. 
    """
    num_weight = 1.0 - cat_weight
    final_fused_risk = (cat_weight * cat_risk_scores) + (num_weight * rcf_risk_scores)
    return final_fused_risk

def optimize_fusion_weights(y_true, cat_risk_scores, rcf_risk_scores, COST_FN=10, COST_FP=2):
    print("\nRunning Cost-Sensitive 2D Optimization (Weight + Threshold)...")
    
    best_weight = 0.5
    best_threshold = 0.5
    lowest_cost = float('inf')
    
    # test weights from 0.0 to 1.0 (in 5% increments)
    weights_to_test = np.linspace(0.0, 1.0, 21) 
    
    # risk thresholds from 0.2 to 0.8 (in 5% increments)
    thresholds_to_test = np.linspace(0.2, 0.8, 13) 
    
    for weight in weights_to_test:
        # Generate the fused risk array for this specific weight
        fused_risk = risk_fusion_scorer(cat_risk_scores, rcf_risk_scores, cat_weight=weight)
        
        for threshold in thresholds_to_test:
            predictions = (fused_risk > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
            current_cost = (fn * COST_FN) + (fp * COST_FP)
            
            if current_cost < lowest_cost:
                lowest_cost = current_cost
                best_weight = weight
                best_threshold = threshold
                
    print(f"Optimal Categorical Weight: {best_weight:.2f}")
    print(f"Optimal Numerical (RCF) Weight: {(1.0 - best_weight):.2f}")
    print(f"Optimal Alert Threshold: {best_threshold:.2f}")
    print(f"Lowest Operational Penalty Score: {lowest_cost}")
    
    return best_weight, best_threshold