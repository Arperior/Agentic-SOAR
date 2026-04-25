import pandas as pd
import numpy as np
from utils import prepare_datasets  # Your existing preprocessing
from scoring import risk_fusion_scorer # Your weight fusion logic

class AgenticSOARPlaybook:
    def __init__(self, cat_model, incident_model, rcf_model, fusion_weight=0.6, threshold=0.7):
        self.cat_model = cat_model         # Binary Classifier
        self.incident_model = incident_model # Multi-class Threat Classifier
        self.rcf_model = rcf_model           # Anomaly Detector
        self.weight = fusion_weight
        self.threshold = threshold

    def execute(self, raw_flow_row):
        """
        The Standard Playbook Loop: 
        Ingest -> Enrich -> Reason -> Action
        """
        # 1. ENRICHMENT: Run your existing ML models
        # (Assuming raw_flow_row is a single-row DataFrame)
        cat_score = self.cat_model.predict_proba(raw_flow_row)[:, 1][0]
        rcf_score = self.rcf_model.score(raw_flow_row.values[0])
        
        # 2. DETECTION: Fuse scores using your scoring.py logic
        fused_risk = (self.weight * cat_score) + ((1 - self.weight) * rcf_score)
        
        if fused_risk < self.threshold:
            return {"status": "SUCCESS", "verdict": "Normal", "risk": fused_risk}

        # 3. INVESTIGATION: If risk is high, the Agent 'Thinks'
        threat_type = self.incident_model.predict(raw_flow_row)[0]
        forensic_narrative = self._generate_forensic_justification(raw_flow_row, threat_type)

        # 4. RESPONSE: Automated Remediation Logic
        remediation = self._get_remediation_action(threat_type, fused_risk)

        return {
            "status": "ALERT",
            "risk_score": round(fused_risk, 3),
            "threat_category": threat_type,
            "forensic_report": forensic_narrative,
            "automated_response": remediation
        }

    def _generate_forensic_justification(self, row, threat):
        """
        Logic based on your Dataset Schema Image.
        The Agent manually inspects features to confirm the ML model's guess.
        """
        reasons = []
        # Logic for DoS (Feature 15: sload, Feature 7: dur)
        if row['sload'].values[0] > 1000000 and row['dur'].values[0] < 0.5:
            reasons.append("High source load (sload) in short duration—highly characteristic of DoS.")
        
        # Logic for Reconnaissance (Feature 41: ct_srv_src)
        if row['ct_srv_src'].values[0] > 10:
            reasons.append(f"Source count for service ({row['ct_srv_src'].values[0]}) is abnormally high.")

        # Logic for Exploit (Feature 10/11: sttl/dttl)
        if row['sttl'].values[0] > 250:
            reasons.append("TTL values are at maximum (254/255), often seen in crafted exploit packets.")

        narrative = f"Agent identified this as {threat}. Evidence: " + " ".join(reasons)
        return narrative

    def _get_remediation_action(self, threat, risk):
        if risk > 0.9:
            return f"CRITICAL: Executing 'iptables -A INPUT -s {threat}_source -j DROP'"
        return "WARNING: Flagged for manual analyst review in SOC Dashboard."

# --- HOW TO RUN THIS IN YOUR PROJECT ---
# 1. Load models using your rcf_model.py and utils.py logic
# 2. Instantiate the Playbook
# playbook = AgenticSOARPlaybook(trained_cat, trained_incident, trained_rcf)
# 3. Feed a row
# alert = playbook.execute(df.iloc[0:1])
# print(alert)