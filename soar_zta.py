import json
import datetime
from os import name

class ZeroTrustSOARAgent:
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        # The Feedback Ledger: Stores dynamic rules based on AI reasoning to prevent future False Positives
        self.dynamic_allowlist = set() 
        self.active_quarantines = set()

    def construct_context(self, event_id, cat_risk, rcf_risk, final_risk, threat_type, telemetry):
        """Packages the mathematical scores and raw telemetry for the AI to read."""
        return {
            "event_id": event_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "ml_risk_profile": {
                "categorical_risk": round(cat_risk, 3),
                "anomaly_risk": round(rcf_risk, 3),
                "fused_risk": round(final_risk, 3)
            },
            "predicted_threat_classification": threat_type,
            "network_telemetry": telemetry.to_dict()
        }

    def evaluate_incident(self, context):
        """
        The AI SOC Analyst. It evaluates the context, actively hunts for False Positives,
        and selects the appropriate Zero Trust playbook.
        """
        system_prompt = """
        You are an autonomous AI SOC Analyst managing a Zero Trust Architecture.
        Your job is to review events flagged by the ML pipeline and select the correct SOAR playbook.
        
        CRITICAL DIRECTIVE: The ML model is mathematically strict and generates False Positives. 
        If the telemetry looks benign (e.g., standard HTTP/FTP traffic with normal states), you must OVERRIDE the ML risk score and issue an "ALLOW" to prevent alert fatigue.
        
        Choose ONE Playbook:
        1. "ALLOW" (Trust Acceptable or verified False Positive)
        2. "STEP_UP_AUTH" (Ambiguous anomaly; require MFA)
        3. "NETWORK_ISOLATION" (Confirmed malicious signature or critical risk)
        
        Respond ONLY with JSON: {"reasoning": "...", "playbook": "...", "is_false_positive": true/false}
        """
        
        # --- SIMULATED LLM REASONING (Replace with actual LLM API call) ---
        telemetry = context["network_telemetry"]
        fused_risk = context["ml_risk_profile"]["fused_risk"]
        
        try:
            # AI Logic: Filtering False Positives
            if fused_risk > 0.4 and context["predicted_threat_classification"] == "Normal":
                if telemetry.get('service') in ['ftp', 'http', 'dns'] and telemetry.get('state') == 'FIN':
                    # The AI recognizes this as a normal connection close, overriding the ML model
                    simulated_response = json.dumps({
                        "reasoning": f"ML flagged anomaly risk ({fused_risk}), but telemetry shows standard {telemetry.get('service')} traffic ending cleanly (FIN). This is a False Positive.",
                        "playbook": "ALLOW",
                        "is_false_positive": True
                    })
                else:
                    simulated_response = json.dumps({
                        "reasoning": "Ambiguous behavioral anomaly detected. Enforcing Zero Trust verification.",
                        "playbook": "STEP_UP_AUTH",
                        "is_false_positive": False
                    })
            elif fused_risk > 0.75 or context["predicted_threat_classification"] != "Normal":
                simulated_response = json.dumps({
                    "reasoning": f"High risk correlation and explicit {context['predicted_threat_classification']} signature match. Isolating immediately.",
                    "playbook": "NETWORK_ISOLATION",
                    "is_false_positive": False
                })
            else:
                simulated_response = json.dumps({
                    "reasoning": "Telemetry within acceptable behavioral baselines.",
                    "playbook": "ALLOW",
                    "is_false_positive": False
                })

            return json.loads(simulated_response)
        
        except Exception as e:
            return {"reasoning": f"LLM Error: {str(e)}", "playbook": "NETWORK_ISOLATION", "is_false_positive": False}

    def execute_playbook(self, event_id, decision, telemetry):
        """Executes the action and triggers the Feedback Loop based on the diagram."""
        playbook = decision["playbook"]
        is_fp = decision.get("is_false_positive", False)
        
        print(f"\n[SOAR] Executing Playbook: {playbook} for Event {event_id}")
        print(f"[SOAR] AI Reasoning: {decision['reasoning']}")
        
        if playbook == "ALLOW":
            if is_fp:
                print(f"[FEEDBACK LOOP] False Positive identified. Updating dynamic baselines to prevent future alerts for this behavior.")
                self.update_feedback_loop("allowlist", telemetry)
            else:
                print("[SOAR] Traffic allowed silently.")
                
        elif playbook == "STEP_UP_AUTH":
            print(f"[SOAR] MFA Challenge issued for protocol {telemetry.get('proto')}. Pending user verification...")
            # If this were a live app, a successful MFA would trigger update_feedback_loop("allowlist", telemetry)
            
        elif playbook == "NETWORK_ISOLATION":
            print(f"[SOAR]  CRITICAL: Connection severed. State logged to quarantine.")
            self.update_feedback_loop("quarantine", telemetry)

    def update_feedback_loop(self, action, telemetry):
        """
        Fulfills the 'Continuous Feedback' path on your architecture diagram.
        Updates internal state so the agent learns over time.
        """
        # Create a behavioral signature (e.g., proto + service + state)
        behavior_signature = f"{telemetry.get('proto')}_{telemetry.get('service')}_{telemetry.get('state')}"
        
        if action == "allowlist":
            self.dynamic_allowlist.add(behavior_signature)
        elif action == "quarantine":
            self.active_quarantines.add(behavior_signature)

'''How to run --->
if name == "__main__":
        # Initialize the new SOAR Agent
    soar_agent = ZeroTrustSOARAgent()

    # Pass an event through
    for i in range(5): 
        row_data = X_train_cat.iloc[i]
        threat_type = incident_agent.predict(row_data)[0][0]
        
        # 1. Package Context
        context = soar_agent.construct_context(
            event_id=i+1, cat_risk=cat_scores_raw[i], rcf_risk=rcf_scores_norm[i], 
            final_risk=final_risk[i], threat_type=threat_type, telemetry=row_data
        )
        
        # 2. AI Evaluation (False Positive Filtering)
        decision = soar_agent.evaluate_incident(context)
        
        # 3. Execute and Update Feedback Loop
        soar_agent.execute_playbook(event_id=i+1, decision=decision, telemetry=row_data)
        '''