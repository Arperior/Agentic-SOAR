"""
soar_zta.py — Zero Trust SOAR Agent

Implements the full pipeline from the flowchart:

  INPUT SOURCES (Endpoint / Network / Identity Logs)
      ↓
  [Receive Security Event]
      ↓
  [Feature Extraction]
      ↓
  [RCF Anomaly Detection]
      ↓
  Is Anomaly Score > Threshold?
      │ No  → [Log benign event] → End
      │ Yes
      ↓
  [Incident Agent → Threat Classification → Risk Agent → Calculate Risk Score]
      ↓
  [Policy Agent]          ← deterministic, no LLM
      ↓
  [Zero Trust Evaluation] ← is_trust_acceptable()
      │ Yes → [ALLOW]
      │ No
      ↓
  [Response Agent]  (LLM evaluate_incident)
      ↓
  [SOAR Playbooks]
      ↓
  [Update Trust Score → Feedback Loop → End]

Changes from previous version
------------------------------
FIX 1 — RCF threshold gate: run() now implements the "Is Anomaly Score >
         Threshold?" diamond at the top of the flowchart. Events that fall
         below the RCF threshold are logged and returned early without
         ever reaching the Policy Agent or LLM.

FIX 2 — Signature consistency: _add_to_allowlist / _add_to_quarantine now
         accept a TrustEvaluation and reuse trust_eval.behavior_signature
         instead of recomputing the signature from raw telemetry. This
         guarantees the memory key always matches what PolicyAgent wrote.

FIX 3 — Removed redundant fast-path checks from evaluate_incident. The
         PolicyAgent already handles quarantine/allowlist signals via
         _apply_memory_signals. Duplicating the logic produced misleading
         audit trails where a full TrustEvaluation was written but the
         actual decision came from a silent memory lookup inside the LLM
         method. PolicyAgent is now the single source of truth for memory
         signals.

FIX 4 — execute_playbook now receives and logs the full TrustEvaluation
         dict so the Splunk ingest step can carry the complete audit trail.
"""

import json
import datetime
import requests
import re
import os
import numpy as np

from policy_agent import PolicyAgent, TrustEvaluation


class _NumpyEncoder(json.JSONEncoder):
    """
    Fallback JSON encoder that handles numpy scalars and arrays.
    Used in evaluate_incident so json.dumps(context) never raises
    TypeError regardless of what telemetry fields contain.
    """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class ZeroTrustSOARAgent:
    def __init__(
        self,
        llm_client=None,
        playbook_file="playbooks.json",
        memory_file="agent_memory.json",
        policy_config_file="policy_config.json",
        threshold_file="Saves/optimal_threshold.json"
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
                "trust_scores": {}
            }
        )

        self.dynamic_allowlist = set(self.memory.get("dynamic_allowlist", []))
        self.active_quarantines = set(self.memory.get("active_quarantines", []))
        self.trust_scores: dict = self.memory.get("trust_scores", {})

        # Policy Agent — deterministic Zero Trust evaluation layer
        self.policy_agent = PolicyAgent(policy_config_file=policy_config_file)

        # Load the OOF-optimised decision threshold produced by find_optimal_threshold().
        # Falls back to 0.5 only if the file is missing (e.g. first run before training).
        self.optimal_threshold = self._load_threshold(threshold_file)

    def _load_threshold(self, threshold_file: str, fallback: float = 0.5) -> float:
        """
        Reads the threshold saved by find_optimal_threshold() in utils.py.
        Uses fallback only when the file doesn't exist yet (pre-training).
        """
        if not os.path.exists(threshold_file):
            print(
                f"[ZeroTrustSOARAgent] WARNING: threshold file '{threshold_file}' not found. "
                f"Falling back to {fallback}. Run training (core.ipynb) to generate it."
            )
            return fallback

        with open(threshold_file, "r") as f:
            record = json.load(f)

        threshold = record.get("optimal_threshold", fallback)
        cost = record.get("soc_cost")
        print(
            f"[ZeroTrustSOARAgent] Loaded optimal threshold: {threshold:.4f} "
            f"(OOF SOC cost={cost}, FN×{record.get('cost_fn')} FP×{record.get('cost_fp')})"
        )
        return threshold

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, event_id, context: dict, optimal_threshold: float = None) -> dict:
        """
        Full pipeline per the flowchart.

        Parameters
        ----------
        event_id          : int | str
        context           : dict — from construct_context()
        optimal_threshold : float | None
            Decision threshold for both the RCF gate and the ZeroTrust
            evaluation diamond. When None (the default), uses the
            OOF-optimised value loaded from Saves/optimal_threshold.json
            at construction time. Pass an explicit value only to override
            for experimentation.

        Returns
        -------
        dict with keys: playbook, reasoning, is_false_positive, trust_eval
        """
        threshold = optimal_threshold if optimal_threshold is not None else self.optimal_threshold
        telemetry = context["network_telemetry"]
        ml_profile = context["ml_risk_profile"]

        # ── Stage 1: RCF Anomaly Gate (flowchart diamond #1) ──────────
        # FIX 1: This gate was missing. Events below the RCF threshold
        # are logged and returned early — Policy Agent and LLM are never
        # invoked, matching the "No → Log" branch in the flowchart.
        rcf_score = ml_profile.get("anomaly_risk", 0.0)
        if rcf_score <= threshold:
            print(
                f"\n[RCF Gate]   anomaly_risk={rcf_score:.4f} ≤ threshold={threshold} "
                f"→ Benign, logging only."
            )
            return self._log_benign_event(event_id, context)

        print(
            f"\n[RCF Gate]   anomaly_risk={rcf_score:.4f} > threshold={threshold} "
            f"→ Escalating to Policy Agent."
        )

        # ── Stage 2: Policy Agent ──────────────────────────────────────
        memory_snapshot = {
            "active_quarantines": self.active_quarantines,
            "dynamic_allowlist": self.dynamic_allowlist
        }
        trust_eval = self.policy_agent.evaluate(context, memory_snapshot)

        print(
            f"\n[PolicyAgent] Trust Score: {trust_eval.trust_score:.4f} "
            f"| Signature: {trust_eval.behavior_signature}"
        )
        if trust_eval.policy_violations:
            print(f"              Violations: {', '.join(trust_eval.policy_violations)}")

        # ── Stage 3: Zero Trust Evaluation (flowchart diamond #2) ─────
        if self.policy_agent.is_trust_acceptable(trust_eval, threshold=threshold):
            print(f"[ZeroTrust]  Trust ACCEPTABLE (≥ {threshold}) → ALLOW fast-path")
            decision = {
                "reasoning": (
                    f"Trust score {trust_eval.trust_score:.4f} meets threshold "
                    f"{threshold}. No escalation required."
                ),
                "playbook": "ALLOW",
                "is_false_positive": False,
                "trust_eval": trust_eval.to_dict()
            }
        else:
            print(f"[ZeroTrust]  Trust UNACCEPTABLE (< {threshold}) → Response Agent")

            # ── Stage 4: Response Agent (LLM) ─────────────────────────
            decision = self.evaluate_incident(context, threshold)
            decision["trust_eval"] = trust_eval.to_dict()

        # ── Stage 5: Execute SOAR Playbook ────────────────────────────
        # FIX 4: Pass trust_eval into execute_playbook for full audit logging.
        self.execute_playbook(event_id, decision, telemetry, trust_eval)

        # ── Stage 6: Update Trust Score + Feedback Loop ───────────────
        self.update_trust_score(trust_eval)
        # FIX 2: Pass trust_eval so feedback loop reuses the pre-computed
        # signature instead of recomputing it from raw telemetry.
        self._trigger_feedback_loop(decision, trust_eval)

        return decision

    # ------------------------------------------------------------------
    # RCF gate helpers
    # ------------------------------------------------------------------

    def _log_benign_event(self, event_id, context: dict) -> dict:
        """
        Handles the 'No' branch of the RCF anomaly gate.
        Logs the event to SIEM and returns a LOG_ONLY decision — no
        Policy Agent, no LLM, no quarantine update.
        """
        print(f"[SOAR] Event {event_id}: RCF score below threshold → LOG_ONLY")
        return {
            "reasoning": (
                "RCF anomaly score did not exceed the detection threshold. "
                "Traffic logged as benign with no further action."
            ),
            "playbook": "LOG_ONLY",
            "is_false_positive": False,
            "trust_eval": None
        }

    # ------------------------------------------------------------------
    # JSON / persistence helpers
    # ------------------------------------------------------------------

    def _load_json(self, filepath, default_data):
        if not os.path.exists(filepath):
            print(f"Creating new configuration file: {filepath}")
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
                    "trust_scores": self.trust_scores
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
            "LOG_ONLY": {
                "description": "RCF score below threshold. Event logged, no action taken.",
                "steps": [
                    {"tool": "Splunk_SIEM", "action": "ingest_telemetry", "index": "network_traffic_benign"}
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

    # ------------------------------------------------------------------
    # Context construction
    # ------------------------------------------------------------------

    @staticmethod
    def _sanitize_for_json(obj):
        """
        Recursively converts numpy scalars and arrays to native Python types
        so the context dict is always safe to pass to json.dumps / the LLM.
        Called once in construct_context so every downstream code path
        (PolicyAgent, evaluate_incident, _save_memory) gets clean types.
        """
        if isinstance(obj, dict):
            return {k: ZeroTrustSOARAgent._sanitize_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [ZeroTrustSOARAgent._sanitize_for_json(v) for v in obj]
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    def construct_context(self, event_id, cat_risk, rcf_risk, final_risk, threat_type, telemetry):
        raw_telemetry = telemetry.to_dict() if hasattr(telemetry, "to_dict") else dict(telemetry)
        return {
            "event_id": int(event_id) if hasattr(event_id, 'item') else event_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "ml_risk_profile": {
                "categorical_risk": round(float(cat_risk), 3),
                "anomaly_risk":     round(float(rcf_risk), 3),
                "fused_risk":       round(float(final_risk), 3)
            },
            "predicted_threat_classification": str(threat_type),
            "network_telemetry": self._sanitize_for_json(raw_telemetry)
        }

    # ------------------------------------------------------------------
    # Response Agent (LLM)
    # ------------------------------------------------------------------

    def evaluate_incident(self, context: dict, optimal_threshold: float) -> dict:
        """
        LLM-based Response Agent (DeepSeek-R1 via Ollama).
        Only called when PolicyAgent deems trust unacceptable.

        FIX 3: Removed the redundant allowlist/quarantine fast-path checks
        that previously duplicated PolicyAgent._apply_memory_signals logic.
        PolicyAgent is now the single source of truth for memory signals.
        Any signature in quarantine or on the allowlist will already have
        had its trust score crushed/boosted by the PolicyAgent, causing the
        ZeroTrust diamond to route correctly before this method is reached.
        """
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
            f"{json.dumps(context, indent=2, cls=_NumpyEncoder)}"
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

    # ------------------------------------------------------------------
    # Playbook execution
    # ------------------------------------------------------------------

    def execute_playbook(
        self,
        event_id,
        decision: dict,
        telemetry,
        trust_eval: TrustEvaluation = None
    ):
        """
        FIX 4: Now accepts trust_eval so the full audit trail (trust score,
        factors, violations) can be forwarded to SIEM ingest steps.
        """
        playbook_name = decision.get("playbook", "NETWORK_ISOLATION")

        print(f"\n[SOAR] Triggering Playbook: {playbook_name} for Event {event_id}")
        print(f"       AI Reasoning: {decision.get('reasoning', 'No reasoning provided.')}")

        if trust_eval is not None:
            print(
                f"       Trust Score: {trust_eval.trust_score:.4f} "
                f"| Factors: {len(trust_eval.factors)} "
                f"| Violations: {len(trust_eval.policy_violations)}"
            )

        playbook_definition = self.playbooks.get(playbook_name)
        if not playbook_definition:
            print(f"       [ERROR] Playbook '{playbook_name}' not found in playbooks.json!")
            return

        for step_num, step in enumerate(playbook_definition["steps"], 1):
            print(f"       ↳ Step {step_num}: Calling {step.get('tool')} API -> {step.get('action')}()")

    # ------------------------------------------------------------------
    # Trust score persistence (Stage 6)
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
    # Feedback loop
    # ------------------------------------------------------------------

    def _trigger_feedback_loop(self, decision: dict, trust_eval: TrustEvaluation):
        """
        Updates allowlist / quarantine based on the final playbook decision.

        FIX 2: Accepts TrustEvaluation instead of raw telemetry so the
        behavior signature is reused directly from trust_eval.behavior_signature
        rather than recomputed. This guarantees the memory key is always
        consistent with what PolicyAgent evaluated.
        """
        playbook_name = decision.get("playbook", "NETWORK_ISOLATION")
        is_fp = decision.get("is_false_positive", False)

        if playbook_name == "ALLOW" and is_fp:
            self._add_to_allowlist(trust_eval)
        elif playbook_name in ["NETWORK_ISOLATION", "RATE_LIMIT_DOS"]:
            self._add_to_quarantine(trust_eval)

    def _add_to_allowlist(self, trust_eval: TrustEvaluation):
        """FIX 2: Uses pre-computed signature from TrustEvaluation."""
        sig = trust_eval.behavior_signature
        if sig not in self.dynamic_allowlist:
            print(f"       [MEMORY] Adding '{sig}' to Allowlist")
            self.dynamic_allowlist.add(sig)
            self._save_memory()

    def _add_to_quarantine(self, trust_eval: TrustEvaluation):
        """FIX 2: Uses pre-computed signature from TrustEvaluation."""
        sig = trust_eval.behavior_signature
        if sig not in self.active_quarantines:
            print(f"       [MEMORY] Adding '{sig}' to Quarantine")
            self.active_quarantines.add(sig)
            self._save_memory()

    # kept for backward compatibility
    def update_feedback_loop(self, action: str, trust_eval: TrustEvaluation):
        if action == "allowlist":
            self._add_to_allowlist(trust_eval)
        elif action == "quarantine":
            self._add_to_quarantine(trust_eval)