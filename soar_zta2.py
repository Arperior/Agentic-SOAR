"""
soar_zta.py — Zero Trust SOAR Agent

Updated to wire in PolicyAgent as the first evaluation stage,
matching the flowchart:

  ML scores
      ↓
  [Policy Agent]          ← deterministic, no LLM
      ↓
  [Zero Trust Evaluation] ← is_trust_acceptable()
      ↓ No                      ↓ Yes
  [Response Agent]         [ALLOW fast-path]
  (LLM evaluate_incident)
      ↓
  [SOAR Playbooks]
      ↓
  [Update Trust Score / Feedback Loop]
"""

import json
import datetime
import requests
import re
import os

from policy_agent import PolicyAgent, TrustEvaluation


class ZeroTrustSOARAgent:
    def __init__(
        self,
        llm_client=None,
        playbook_file="playbooks.json",
        memory_file="agent_memory.json",
        policy_config_file="policy_config.json"
    ):
        self.llm_client = llm_client
        self.playbook_file = playbook_file
        self.memory_file = memory_file

        self.playbooks = self._load_json(
            filepath=self.playbook_file,
            default_data=self._get_default_playbooks()
        )

        # Agent memory: allowlist, quarantines, and trust score history
        self.memory = self._load_json(
            filepath=self.memory_file,
            default_data={
                "dynamic_allowlist": [],
                "active_quarantines": [],
                "trust_scores": {}       # NEW: signature → last trust score
            }
        )

        self.dynamic_allowlist = set(self.memory.get("dynamic_allowlist", []))
        self.active_quarantines = set(self.memory.get("active_quarantines", []))
        self.trust_scores: dict = self.memory.get("trust_scores", {})

        # Policy Agent — deterministic Zero Trust evaluation layer
        self.policy_agent = PolicyAgent(policy_config_file=policy_config_file)

    # ------------------------------------------------------------------
    # Main entry point (replaces calling evaluate_incident directly)
    # ------------------------------------------------------------------

    def run(self, event_id, context: dict, optimal_threshold: float = 0.5) -> dict:
        """
        Full pipeline per the flowchart:
          Policy Agent → Zero Trust Evaluation → (ALLOW | Response Agent)
          → SOAR Playbook → Update Trust Score → Feedback Loop

        Parameters
        ----------
        event_id : int | str
        context  : dict — from construct_context()
        optimal_threshold : float — trust score gate (default 0.5)

        Returns
        -------
        dict with keys: playbook, reasoning, is_false_positive, trust_eval
        """
        telemetry = context["network_telemetry"]

        # ── Stage 1: Policy Agent ──────────────────────────────────────
        memory_snapshot = {
            "active_quarantines": self.active_quarantines,
            "dynamic_allowlist": self.dynamic_allowlist
        }
        trust_eval = self.policy_agent.evaluate(context, memory_snapshot)

        print(f"\n[PolicyAgent] Trust Score: {trust_eval.trust_score:.4f} "
              f"| Signature: {trust_eval.behavior_signature}")
        if trust_eval.policy_violations:
            print(f"             Violations: {', '.join(trust_eval.policy_violations)}")

        # ── Stage 2: Zero Trust Evaluation (the diamond) ──────────────
        if self.policy_agent.is_trust_acceptable(trust_eval, threshold=optimal_threshold):
            print(f"[ZeroTrust]  Trust ACCEPTABLE (≥ {optimal_threshold}) → ALLOW fast-path")
            decision = {
                "reasoning": (
                    f"Trust score {trust_eval.trust_score:.4f} meets threshold "
                    f"{optimal_threshold}. No escalation required."
                ),
                "playbook": "ALLOW",
                "is_false_positive": False,
                "trust_eval": trust_eval.to_dict()
            }
        else:
            print(f"[ZeroTrust]  Trust UNACCEPTABLE (< {optimal_threshold}) → Response Agent")

            # ── Stage 3: Response Agent (LLM) ─────────────────────────
            decision = self.evaluate_incident(context, optimal_threshold)
            decision["trust_eval"] = trust_eval.to_dict()

        # ── Stage 4: Execute SOAR Playbook ────────────────────────────
        self.execute_playbook(event_id, decision, telemetry)

        # ── Stage 5: Update Trust Score + Feedback Loop ───────────────
        self.update_trust_score(trust_eval)
        self._trigger_feedback_loop(decision, telemetry, trust_eval)

        return decision

    # ------------------------------------------------------------------
    # Existing methods (unchanged logic, minor additions noted)
    # ------------------------------------------------------------------

    def _load_json(self, filepath, default_data):
        if not os.path.exists(filepath):
            print(f"⚙️ Creating new configuration file: {filepath}")
            with open(filepath, "w") as f:
                json.dump(default_data, f, indent=4)
            return default_data
        with open(filepath, "r") as f:
            return json.load(f)

    def _save_memory(self):
        with open(self.memory_file, "w") as f:
            json.dump(
                {
                    "dynamic_allowlist": list(self.dynamic_allowlist),
                    "active_quarantines": list(self.active_quarantines),
                    "trust_scores": self.trust_scores     # NEW: persist trust scores
                },
                f, indent=4
            )

    def _get_default_playbooks(self):
        return {
            "ALLOW": {
                "description": "Standard benign traffic or AI-verified False Positive.",
                "steps": [
                    {"tool": "Splunk_SIEM", "action": "ingest_telemetry", "index": "network_traffic_allowed"},
                    {"tool": "Metrics_Dashboard", "action": "increment_fp_counter", "status": "success"}
                ]
            },
            "STEP_UP_AUTH": {
                "description": "Ambiguous behaviour. Enforce Zero Trust Identity Verification.",
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
                "description": "Volumetric anomaly detected (DoS). Throttle traffic.",
                "steps": [
                    {"tool": "Cloudflare_WAF", "action": "enable_under_attack_mode"},
                    {"tool": "F5_Load_Balancer", "action": "throttle_source_ip", "limit": "100_req_per_min"},
                    {"tool": "Splunk_SIEM", "action": "create_notable_event", "severity": "High"}
                ]
            }
        }

    def construct_context(self, event_id, cat_risk, rcf_risk, final_risk, threat_type, telemetry):
        return {
            "event_id": event_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "ml_risk_profile": {
                "categorical_risk": round(cat_risk, 3),
                "anomaly_risk": round(rcf_risk, 3),
                "fused_risk": round(final_risk, 3)
            },
            "predicted_threat_classification": threat_type,
            "network_telemetry": telemetry.to_dict() if hasattr(telemetry, "to_dict") else dict(telemetry)
        }

    def evaluate_incident(self, context: dict, optimal_threshold: float) -> dict:
        """
        LLM-based Response Agent (DeepSeek-R1 via Ollama).
        Only called when PolicyAgent deems trust unacceptable.
        Fast-path memory checks are preserved but now secondary to the
        PolicyAgent's deterministic verdict.
        """
        telemetry = context["network_telemetry"]
        behavior_signature = (
            f"{telemetry.get('proto')}_{telemetry.get('service')}_{telemetry.get('state')}"
        )

        # Fast-path memory checks (retained for speed on known signatures)
        if behavior_signature in self.dynamic_allowlist:
            print(f"⚡ [FAST-PATH] '{behavior_signature}' in AI Allowlist. Bypassing LLM...")
            return {
                "reasoning": f"Auto-Allowed: '{behavior_signature}' was previously verified as FP.",
                "playbook": "ALLOW",
                "is_false_positive": False
            }

        if behavior_signature in self.active_quarantines:
            print(f"🛡️ [FAST-PATH] '{behavior_signature}' in Quarantine. Bypassing LLM...")
            return {
                "reasoning": f"Auto-Blocked: '{behavior_signature}' is a known hostile signature.",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False
            }

        mfa_trigger = optimal_threshold
        isolation_trigger = min(optimal_threshold + 0.35, 0.75)

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
        Output ONLY a raw JSON object. No markdown. Reasoning: 1-2 concise technical sentences.

        {{
            "reasoning": "...",
            "playbook": "ALLOW" | "STEP_UP_AUTH" | "NETWORK_ISOLATION" | "RATE_LIMIT_DOS",
            "is_false_positive": true | false
        }}
        """

        user_prompt = (
            f"Evaluate this context strictly according to the rules and output ONLY JSON:\n"
            f"{json.dumps(context, indent=2)}"
        )

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "deepseek-r1:8b",
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {"temperature": 0.0}
        }

        try:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            raw_text = response.json()["response"]

            cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
            for fence in ("```json", "```"):
                if cleaned.startswith(fence):
                    cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            return json.loads(cleaned)

        except Exception as e:
            print(f"[ERROR] Local LLM Evaluation Failed: {e}")
            return {
                "reasoning": "Fail-safe triggered due to LLM evaluation timeout/error.",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False
            }

    def execute_playbook(self, event_id, decision: dict, telemetry):
        playbook_name = decision.get("playbook", "NETWORK_ISOLATION")
        is_fp = decision.get("is_false_positive", False)

        print(f"\n[SOAR] Triggering Playbook: {playbook_name} for Event {event_id}")
        print(f"       AI Reasoning: {decision.get('reasoning', 'No reasoning provided.')}")

        playbook_definition = self.playbooks.get(playbook_name)
        if not playbook_definition:
            print(f"       [ERROR] Playbook '{playbook_name}' not found in playbooks.json!")
            return

        for step_num, step in enumerate(playbook_definition["steps"], 1):
            print(f"       ↳ Step {step_num}: Calling {step.get('tool')} API -> {step.get('action')}()")

    # ------------------------------------------------------------------
    # NEW: Trust Score update (Stage 5 of flowchart)
    # ------------------------------------------------------------------

    def update_trust_score(self, trust_eval: TrustEvaluation):
        """
        Persists the latest trust score for a behavior signature.
        Over time this builds a historical trust profile per signature
        that the PolicyAgent can use for trending (future extension).
        """
        sig = trust_eval.behavior_signature
        previous = self.trust_scores.get(sig)

        self.trust_scores[sig] = {
            "latest": trust_eval.trust_score,
            "previous": previous["latest"] if isinstance(previous, dict) else previous,
            "evaluated_at": trust_eval.evaluated_at,
            "violations": trust_eval.policy_violations
        }

        delta_str = ""
        if isinstance(previous, dict) and previous.get("latest") is not None:
            delta = trust_eval.trust_score - previous["latest"]
            delta_str = f" (Δ {delta:+.4f} from previous)"

        print(f"       [TrustScore] '{sig}' → {trust_eval.trust_score:.4f}{delta_str}")
        self._save_memory()

    # ------------------------------------------------------------------
    # Feedback loop (renamed from update_feedback_loop, logic preserved)
    # ------------------------------------------------------------------

    def _trigger_feedback_loop(self, decision: dict, telemetry, trust_eval: TrustEvaluation):
        """
        Updates allowlist / quarantine based on the final playbook decision.
        Replaces the old update_feedback_loop; same behaviour, cleaner name.
        """
        playbook_name = decision.get("playbook", "NETWORK_ISOLATION")
        is_fp = decision.get("is_false_positive", False)

        if playbook_name == "ALLOW" and is_fp:
            self._add_to_allowlist(telemetry)
        elif playbook_name in ["NETWORK_ISOLATION", "RATE_LIMIT_DOS"]:
            self._add_to_quarantine(telemetry)

    def _add_to_allowlist(self, telemetry):
        sig = self.policy_agent._get_signature(
            telemetry.to_dict() if hasattr(telemetry, "to_dict") else dict(telemetry)
        )
        if sig not in self.dynamic_allowlist:
            print(f"       [MEMORY] Adding '{sig}' to Allowlist")
            self.dynamic_allowlist.add(sig)
            self._save_memory()

    def _add_to_quarantine(self, telemetry):
        sig = self.policy_agent._get_signature(
            telemetry.to_dict() if hasattr(telemetry, "to_dict") else dict(telemetry)
        )
        if sig not in self.active_quarantines:
            print(f"       [MEMORY] Adding '{sig}' to Quarantine")
            self.active_quarantines.add(sig)
            self._save_memory()

    # kept for backward compatibility with old commented demo code
    def update_feedback_loop(self, action: str, telemetry):
        if action == "allowlist":
            self._add_to_allowlist(telemetry)
        elif action == "quarantine":
            self._add_to_quarantine(telemetry)
