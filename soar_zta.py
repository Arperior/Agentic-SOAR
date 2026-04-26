import json
import datetime
import requests
import re
import os

class ZeroTrustSOARAgent:
    def __init__(self, llm_client=None, playbook_file="playbooks.json", memory_file="agent_memory.json"):
        self.llm_client = llm_client
        self.playbook_file = playbook_file
        self.memory_file = memory_file
        
        self.playbooks = self._load_json(
            filepath=self.playbook_file, 
            default_data=self._get_default_playbooks()
        )
        
        # Initialize the Agent's Memory (The Feedback Loop)
        self.memory = self._load_json(
            filepath=self.memory_file, 
            default_data={"dynamic_allowlist": [], "active_quarantines": []}
        )
        
        self.dynamic_allowlist = set(self.memory.get("dynamic_allowlist", []))
        self.active_quarantines = set(self.memory.get("active_quarantines", []))

    def _load_json(self, filepath, default_data):
        """Utility to safely load JSON or create it if missing."""
        if not os.path.exists(filepath):
            print(f"⚙️ Creating new configuration file: {filepath}")
            with open(filepath, 'w') as f:
                json.dump(default_data, f, indent=4)
            return default_data
            
        with open(filepath, 'r') as f:
            return json.load(f)
            
    def _save_memory(self):
        """Commits the agent's updated learning state back to the JSON file."""
        with open(self.memory_file, 'w') as f:
            json.dump({
                "dynamic_allowlist": list(self.dynamic_allowlist),
                "active_quarantines": list(self.active_quarantines)
            }, f, indent=4)

    def _get_default_playbooks(self):
        """Provides the enterprise baseline if playbooks.json is missing."""
        return {
            "ALLOW": {
                "description": "Standard benign traffic or AI-verified False Positive.",
                "steps": [
                    {"tool": "Splunk_SIEM", "action": "ingest_telemetry", "index": "network_traffic_allowed"},
                    {"tool": "Metrics_Dashboard", "action": "increment_fp_counter", "status": "success"}
                ]
            },
            "STEP_UP_AUTH": {
                "description": "Ambiguous behavior. Enforce Zero Trust Identity Verification.",
                "steps": [
                    {"tool": "Okta_IAM", "action": "trigger_push_mfa", "timeout": 60},
                    {"tool": "Splunk_SIEM", "action": "create_notable_event", "severity": "Medium"},
                    {"tool": "Jira_ServiceDesk", "action": "open_ticket", "assignee": "Tier_1_SOC"}
                ]
            },
            "NETWORK_ISOLATION": {
                "description": "Confirmed malicious signature or critical risk. Immediate host containment.",
                "steps": [
                    {"tool": "VirusTotal_API", "action": "query_ip_reputation"},
                    {"tool": "Palo_Alto_Firewall", "action": "block_src_ip", "duration": "permanent"},
                    {"tool": "CrowdStrike_EDR", "action": "isolate_host_network"},
                    {"tool": "Active_Directory", "action": "revoke_session_tokens"},
                    {"tool": "PagerDuty", "action": "page_on_call_engineer", "priority": "CRITICAL"}
                ]
            },
            "RATE_LIMIT_DOS": {
                "description": "Volumetric anomaly detected (Denial of Service). Throttle traffic.",
                "steps": [
                    {"tool": "Cloudflare_WAF", "action": "enable_under_attack_mode"},
                    {"tool": "F5_Load_Balancer", "action": "throttle_source_ip", "limit": "100_req_per_min"},
                    {"tool": "Splunk_SIEM", "action": "create_notable_event", "severity": "High"}
                ]
            }
        }

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

    def evaluate_incident(self, context, optimal_threshold):
        """
        The AI SOC Analyst powered by DeepSeek-R1.
        Uses dynamic thresholds calculated by the Cost-Sensitive Optimizer.
        """
        telemetry = context["network_telemetry"]
        behavior_signature = f"{telemetry.get('proto')}_{telemetry.get('service')}_{telemetry.get('state')}"
        
        # --- FAST-PATH MEMORY CHECK ---
        if behavior_signature in self.dynamic_allowlist:
            print(f"⚡ [FAST-PATH] Behavior '{behavior_signature}' recognized in AI Allowlist. Bypassing LLM...")
            return {
                "reasoning": f"Auto-Allowed: '{behavior_signature}' was previously verified as a False Positive.",
                "playbook": "ALLOW",
                "is_false_positive": False # It is now treated as standard normal traffic
            }
            
        if behavior_signature in self.active_quarantines:
            print(f"🛡️ [FAST-PATH] Behavior '{behavior_signature}' recognized in Quarantine. Bypassing LLM...")
            return {
                "reasoning": f"Auto-Blocked: '{behavior_signature}' is a known hostile signature.",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False
            }
        
        # 1. Define the dynamic boundaries
        mfa_trigger = optimal_threshold
        isolation_trigger = min(optimal_threshold + 0.35, 0.75) 

        # 2. Inject them dynamically into the System Prompt
        system_prompt = f"""
        You are an AI SOC Analyst evaluating network telemetry and ML risk scores.
        Your goal is to enforce Zero Trust while preventing alert fatigue.

        EVALUATION RULES:
        1. IF 'fused_risk' <= {mfa_trigger}: Playbook is 'ALLOW' (is_false_positive: false).
        2. IF 'predicted_threat_classification' == "DoS": Playbook is 'RATE_LIMIT_DOS' (is_false_positive: false).
        3. IF 'fused_risk' > {isolation_trigger} OR ('predicted_threat_classification' != "Normal" AND 'predicted_threat_classification' != "DoS"): Playbook is 'NETWORK_ISOLATION' (is_false_positive: false).
        4. IF 'fused_risk' > {mfa_trigger} AND 'predicted_threat_classification' == "Normal":
           - Inspect 'network_telemetry'. If traffic is standard and benign (e.g., 'service' is http/ftp/dns AND 'state' indicates a clean close like FIN), OVERRIDE the ML: Playbook is 'ALLOW' (is_false_positive: true).
           - If traffic behaves ambiguously: Playbook is 'STEP_UP_AUTH' (is_false_positive: false).

        OUTPUT CONSTRAINTS:
        Output ONLY a raw JSON object. Do not include markdown formatting. Keep reasoning to 1-2 concise technical sentences.
        
        {{
            "reasoning": "...",
            "playbook": "ALLOW" | "STEP_UP_AUTH" | "NETWORK_ISOLATION" | "RATE_LIMIT_DOS",
            "is_false_positive": true | false
        }}
        """
        
        user_prompt = f"Evaluate this context strictly according to the rules and output ONLY JSON:\n{json.dumps(context, indent=2)}"
        
        # --- LOCAL LLM API CALL (OLLAMA) ---
        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "deepseek-r1:8b",
            "system": system_prompt, 
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0  
            }
        }
        
        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            
            raw_text = response.json()["response"]
            
            # --- DEEPSEEK-R1 CLEANUP LOGIC ---
            cleaned_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL).strip()
            
            if cleaned_text.startswith("```json"):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith("```"):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith("```"):
                cleaned_text = cleaned_text[:-3]
                
            cleaned_text = cleaned_text.strip()
            
            decision = json.loads(cleaned_text)
            return decision
            
        except Exception as e:
            print(f"[ERROR] Local LLM Evaluation Failed: {e}")
            return {
                "reasoning": "Fail-safe triggered due to local LLM evaluation timeout/error.",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False
            }

    def execute_playbook(self, event_id, decision, telemetry):
        """Fetches the playbook definition from JSON and executes its steps."""
        playbook_name = decision.get("playbook", "NETWORK_ISOLATION")
        is_fp = decision.get("is_false_positive", False)
        
        print(f"\n[SOAR] Triggering Playbook: {playbook_name} for Event {event_id}")
        print(f"       AI Reasoning: {decision.get('reasoning', 'No reasoning provided.')}")
        
        # Look up the playbook dynamically from the loaded JSON data
        playbook_definition = self.playbooks.get(playbook_name)
        
        if not playbook_definition:
            print(f"       [ERROR] Playbook '{playbook_name}' not found in playbooks.json!")
            return

        # Simulate executing the enterprise APIs
        for step_num, step in enumerate(playbook_definition["steps"], 1):
            tool = step.get("tool")
            action = step.get("action")
            print(f"       ↳ Step {step_num}: Calling {tool} API -> {action}()")
        
        # Trigger the feedback loop to save state dynamically
        if playbook_name == "ALLOW" and is_fp:
            self.update_feedback_loop("allowlist", telemetry)
        elif playbook_name in ["NETWORK_ISOLATION", "RATE_LIMIT_DOS"]:
            self.update_feedback_loop("quarantine", telemetry)

    def update_feedback_loop(self, action, telemetry):
        """Updates internal state and writes to the JSON file so the agent learns over time."""
        behavior_signature = f"{telemetry.get('proto')}_{telemetry.get('service')}_{telemetry.get('state')}"
        
        if action == "allowlist" and behavior_signature not in self.dynamic_allowlist:
            print(f"       [MEMORY] Saving '{behavior_signature}' to agent_memory.json (Allowlist)")
            self.dynamic_allowlist.add(behavior_signature)
            self._save_memory() # Persist to disk dynamically
            
        elif action == "quarantine" and behavior_signature not in self.active_quarantines:
            print(f"       [MEMORY] Saving '{behavior_signature}' to agent_memory.json (Quarantine)")
            self.active_quarantines.add(behavior_signature)
            self._save_memory() # Persist to disk dynamically

'''
# 1. Initialize the Autonomous Zero Trust Agent
print("\nInitializing Autonomous Zero Trust SOAR Agent...")
soar_agent = ZeroTrustSOARAgent()

print("\n--- AGENTIC SOAR LIVE EXECUTION LOG ---")

# 2. Stream the first 5 events from the validation set into the Agent
for i in range(5):
    # Retrieve the exact network telemetry row for this validation event
    row_idx = idx_val[i]
    telemetry_row = X_train_cat.iloc[row_idx]
    
    # Get the specific threat classification from the Incident Agent
    threat_type = incident_model.predict(telemetry_row)[0][0]
    
    # Package the mathematical context
    context = soar_agent.construct_context(
        event_id=i+1,
        cat_risk=X_meta_val[i][0],
        rcf_risk=X_meta_val[i][1],
        final_risk=val_final_risk[i],
        threat_type=threat_type,
        telemetry=telemetry_row
    )
    
    # 3. AI Evaluation: Because the Meta-Learner is cost-adjusted, the boundary is 0.5
    decision = soar_agent.evaluate_incident(context, optimal_threshold=0.5)
    
    # 4. Execute Playbook and Update Feedback Memory
    soar_agent.execute_playbook(event_id=i+1, decision=decision, telemetry=telemetry_row)
'''