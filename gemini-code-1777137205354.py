import pandas as pd
from utils import preprocess_data # Using your existing work
from scoring import model_predict # Using your existing work

class AgenticSOC:
    def __init__(self, model_path):
        self.model_path = model_path
        # Map features from your uploaded image to "Investigation Skills"
        self.investigation_skills = {
            "load_analysis": ["sload", "dload", "sbytes", "dbytes"],
            "timing_analysis": ["dur", "sjit", "djit", "sttl"],
            "connection_behavior": ["ct_srv_src", "ct_dst_ltm", "ct_src_dport_ltm"]
        }

    def run_playbook(self, raw_flow):
        """
        Standard Organizational Playbook:
        Detect -> Investigate -> Justify -> Respond
        """
        # STEP 1: DETECTION (Using your current scoring.py logic)
        processed_flow = preprocess_data(raw_flow)
        risk_score, initial_pred = model_predict(processed_flow, self.model_path)
        
        if risk_score < 0.7:
            return {"status": "Clear", "score": risk_score}

        # STEP 2: AGENTIC INVESTIGATION (The 'Agentic' Part)
        # The Agent 'looks' at specific feature groups to explain the score
        investigation_report = self._perform_forensics(raw_flow)
        
        # STEP 3: REASONING & OUTPUT
        # This string is what you'd feed to an LLM to generate a final SOC report
        final_report = {
            "alert_id": "SOC-UNSW-001",
            "model_prediction": initial_pred,
            "confidence": risk_score,
            "forensic_evidence": investigation_report,
            "playbook_verdict": "Escalate to Human" if risk_score > 0.9 else "Monitor"
        }
        
        return final_report

    def _perform_forensics(self, flow):
        # Reasoning logic based on the feature descriptions in your image
        evidence = []
        if flow['sload'] > 1000000 and flow['dur'] < 0.1:
            evidence.append("High-speed load in short duration: Signature of DoS/Fuzzer.")
        if flow['ct_srv_src'] > 5:
            evidence.append("Multiple connections to same service: Signature of Reconnaissance.")
        return evidence

# Integration Example
# agent = AgenticSOC("path/to/your/model.pkl")
# report = agent.run_playbook(new_network_data)