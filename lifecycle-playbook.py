import pandas as pd
import numpy as np
import rrcf
import tqdm
from catboost import CatBoostClassifier

# ==========================================
# PART 1: ANOMALY DETECTION ENGINE (RCF)
# ==========================================
class RCF:
    def __init__(self, num_trees=40, tree_size=256, seed=0, warmup=500, smoothing_window=20):
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.rng = np.random.default_rng(seed)
        self.forest = [rrcf.RCTree() for _ in range(num_trees)]
        self.index = 0
        self.warmup = warmup
        self.smoothing_window = smoothing_window
        self.history = []

    def score(self, x):
        scores = []
        for tree in self.forest:
            if len(tree.leaves) >= self.tree_size:
                drop_index = self.rng.choice(list(tree.leaves.keys()))
                tree.forget_point(drop_index)
            tree.insert_point(x, index=self.index)
            scores.append(tree.codisp(self.index))
        
        self.index += 1
        raw_score = float(np.mean(scores))
        normalized = 1.0 / (1.0 + np.exp(-np.log1p(raw_score)))
        
        self.history.append(normalized)
        if len(self.history) > self.smoothing_window:
            self.history.pop(0)
        return float(np.mean(self.history))

# ==========================================
# PART 2: THE AGENTIC PLAYBOOK LOGIC
# ==========================================
class AgenticSOARPlaybook:
    def __init__(self, cat_model, incident_model, rcf_model, fusion_weight=0.6, threshold=0.7):
        """
        cat_model: Binary Classifier (Normal vs Attack)
        incident_model: Multi-class Classifier (Specific Attack Category)
        rcf_model: Streaming Anomaly Detector
        """
        self.cat_model = cat_model
        self.incident_model = incident_model
        self.rcf_model = rcf_model
        self.weight = fusion_weight
        self.threshold = threshold

    def execute_investigation(self, raw_row):
        """
        Standard Playbook Workflow: Detect -> Classify -> Justify -> Respond
        """
        # 1. ANALYTICS STEP (Detection)
        cat_score = self.cat_model.predict_proba(raw_row)[:, 1][0]
        rcf_score = self.rcf_model.score(raw_row.values[0])
        
        # Risk Fusion (from your scoring.py logic)
        fused_risk = (self.weight * cat_score) + ((1 - self.weight) * rcf_score)
        
        if fused_risk < self.threshold:
            return {"status": "CLEAN", "risk": round(fused_risk, 3), "verdict": "Normal Traffic"}

        # 2. INTELLIGENCE STEP (Forensics)
        threat_type = self.incident_model.predict(raw_row)[0]
        if isinstance(threat_type, np.ndarray): threat_type = threat_type[0]
        
        forensic_report = self._generate_forensics(raw_row, threat_type)

        # 3. RESPONSE STEP (Automation)
        action = "BLOCK_IP" if fused_risk > 0.85 else "LOG_AND_MONITOR"

        return {
            "status": "ALERT_TRIGGERED",
            "risk_score": round(fused_risk, 3),
            "threat_category": threat_type,
            "agent_reasoning": forensic_report,
            "automated_response": action
        }

    def _generate_forensics(self, row, threat):
        """
        Heuristic rules mapped to the UNSW-NB15 dataset schema
        """
        reasons = []
        # Helper to get value
        val = lambda col: row[col].values[0]

        # DOS & FUZZERS
        if threat in ['DoS', 'Fuzzers']:
            if val('sload') > 500000:
                reasons.append(f"High Source Load detected: {val('sload'):.0f} bps.")
            if val('dur') < 0.1:
                reasons.append("Extreme short-duration burst behavior.")

        # RECONNAISSANCE
        elif threat == 'Reconnaissance':
            if val('ct_srv_src') > 10:
                reasons.append(f"Port Scanning suspected: {val('ct_srv_src')} service hits.")
            if val('is_sm_ips_ports') == 1:
                reasons.append("Identical source/destination detected (LANS attack pattern).")

        # EXPLOITS & SHELLCODE
        elif threat in ['Exploits', 'Shellcode']:
            if val('sttl') >= 252:
                reasons.append(f"Anomalous TTL ({val('sttl')}): Potential OS-fingerprinting bypass.")
            if val('smeansz') > 500:
                reasons.append("Large payload size detected in control packet.")

        # BACKDOORS & WORMS
        elif threat in ['Backdoor', 'Worms']:
            if val('ct_src_dport_ltm') > 5:
                reasons.append("Lateral movement pattern: rapid destination port hopping.")
            if val('sloss') > 0:
                reasons.append("Tunnel instability/Packet loss detected.")

        return f"Forensic analysis for {threat}: " + (" ".join(reasons) if reasons else "General statistical anomaly.")

# ==========================================
# PART 3: BOOTSTRAP & RUN
# ==========================================
def run_agentic_system(train_path):
    print("Initializing Agentic System...")
    
    # Preprocessing (Simplified from your utils.py)
    df = pd.read_csv(train_path)
    X = df.drop(['label', 'attack_cat', 'id'], axis=1, errors='ignore')
    y_binary = df['label']
    y_multi = df['attack_cat'].fillna('Normal')
    
    # Identify Cat Features
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    X[cat_features] = X[cat_features].fillna('unknown').astype(str)
    
    # 1. Train Models
    print("Training Decision Engines...")
    bin_model = CatBoostClassifier(iterations=50, cat_features=cat_features, verbose=0).fit(X, y_binary)
    inc_model = CatBoostClassifier(iterations=50, cat_features=cat_features, verbose=0).fit(X, y_multi)
    
    # 2. Setup Playbook
    my_rcf = RCF()
    playbook = AgenticSOARPlaybook(bin_model, inc_model, my_rcf)
    
    # 3. Process Sample Row
    sample_row = X.iloc[0:1] # Test with one row
    result = playbook.execute_investigation(sample_row)
    
    print("\n--- AGENT OUTPUT ---")
    import json
    print(json.dumps(result, indent=4))

# To run:
# run_agentic_system('unsw_train.csv')