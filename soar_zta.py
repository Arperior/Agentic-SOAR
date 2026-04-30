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

        self.trust_threshold = self.policy_agent.config.get("trust_score_threshold", 0.40)

    def _load_threshold(self, threshold_file: str, fallback: float = 0.5) -> float:
        if not os.path.exists(threshold_file):
            return fallback

        with open(threshold_file, "r") as f:
            record = json.load(f)

        # FIX: Catch the silent fallback flag and warn the SOC
        if record.get("is_fallback", False):
            print(
                f"\n[CRITICAL WARNING] The optimal threshold file indicates a fallback to 0.5. "
                f"The ML pipeline failed to meet the minimum precision floor during training. "
                f"Retraining is highly recommended!\n"
            )

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

        # ── Stage 1: Fused ML Gate (flowchart diamond #1) ──────────
        # Uses the ML-optimised threshold — this is the correct domain for it.
        # Events below the fused threshold are logged and returned early.
        fused_score = ml_profile.get("fused_risk", 0.0)
        
        if fused_score <= threshold:
            print(
                f"\n[ML Gate]   fused_risk={fused_score:.4f} ≤ threshold={threshold:.4f} "
                f"→ Benign, logging only."
            )
            return self._log_benign_event(event_id, context)

        print(
            f"\n[ML Gate]   fused_risk={fused_score:.4f} > threshold={threshold:.4f} "
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
        is_trust_ok = self.policy_agent.is_trust_acceptable(trust_eval, threshold=self.trust_threshold)
        
        # GREY ZONE GUARDRAIL: Never fast-path if the ML is highly confident it is an attack
        grey_zone_limit = 0.85 
        is_below_grey_zone = fused_score < grey_zone_limit

        # Only flag as a false positive (and thus allowlist the signature) when
        # the fused score is well below the detection boundary — not just anywhere
        # under the grey-zone ceiling.  A score that barely clears the ML threshold
        # but still passes the trust check should be ALLOWed without being memorised,
        # because its risk profile could easily flip on the next encounter.
        LOW_RISK_CEILING = threshold * 0.75  # e.g. 0.558 at default threshold 0.7448
        is_genuinely_low_risk = fused_score < LOW_RISK_CEILING

        if is_trust_ok and is_below_grey_zone:
            print(f"[ZeroTrust]  Trust ACCEPTABLE (≥ {self.trust_threshold}) AND Risk < 0.85 → ALLOW fast-path")
            decision = {
                "reasoning": (
                    f"Trust score {trust_eval.trust_score:.4f} meets threshold "
                    f"{self.trust_threshold}. No escalation required."
                ),
                "playbook": "ALLOW",
                # Mark as FP (and allowlist) only when fused_risk is comfortably
                # below the detection boundary.  Mid-range scores that happen to
                # pass the trust check are allowed but NOT memorised — their risk
                # profile is too ambiguous for a permanent fast-path exemption.
                "is_false_positive": is_genuinely_low_risk,
                "trust_eval": trust_eval.to_dict()
            }
        else:
            if is_trust_ok and not is_below_grey_zone:
                print(f"[ZeroTrust]  Trust ACCEPTABLE but Risk {fused_score:.4f} ≥ 0.85 → Vetoing fast-path, escalating to Response Agent")
            else:
                print(f"[ZeroTrust]  Trust UNACCEPTABLE (< {self.trust_threshold}) → Response Agent")

            # ── Stage 4: Response Agent (LLM) ─────────────────────────
            # FIX (Quarantine short-circuit): If this signature is already in
            # active_quarantines, skip the LLM entirely — we already know it's
            # malicious. This prevents hammering Ollama with repeat events for
            # the same bad signature (a major source of LLM timeout errors).
            if trust_eval.behavior_signature in self.active_quarantines:
                print(f"[ZeroTrust]  Signature '{trust_eval.behavior_signature}' already quarantined "
                      f"→ NETWORK_ISOLATION (LLM skipped)")
                decision = {
                    "reasoning": (
                        f"Signature '{trust_eval.behavior_signature}' is in active quarantine. "
                        f"Immediate isolation without LLM re-evaluation."
                    ),
                    "playbook": "NETWORK_ISOLATION",
                    "is_false_positive": False,
                    "is_llm_failsafe": False
                }
            else:
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
            "routing_rules": [
                "1. IF 'fused_risk' <= {mfa_trigger} AND 'predicted_threat_classification' == 'Normal': Playbook is 'ALLOW' (is_false_positive: true).",
                "2. IF 'fused_risk' <= {mfa_trigger} AND 'predicted_threat_classification' != 'Normal': Playbook is 'ALLOW' (is_false_positive: false).",
                "3. IF 'predicted_threat_classification' == 'DoS': Playbook is 'RATE_LIMIT_DOS' (is_false_positive: false).",
                "4. IF 'predicted_threat_classification' == 'Shellcode' AND 'proto' == 'udp' AND 'dpkts' == 0 AND 'fused_risk' < {mfa_trigger}: Playbook is 'ALLOW' (is_false_positive: true). NOTE: This rule applies ONLY to Shellcode misclassifications with low fused risk. Generic, Reconnaissance, or other non-Normal threat classes with dpkts=0 are NOT covered by this rule.",
                "5. IF 'fused_risk' > {isolation_trigger}: Playbook is 'NETWORK_ISOLATION' (is_false_positive: false).",
                "6. IF 'predicted_threat_classification' != 'Normal' AND 'predicted_threat_classification' != 'DoS': Playbook is 'NETWORK_ISOLATION' (is_false_positive: false).",
                "7. IF 'fused_risk' > {mfa_trigger} AND 'predicted_threat_classification' == 'Normal': Inspect telemetry. If standard benign traffic, 'ALLOW'. Otherwise, 'STEP_UP_AUTH'."
            ],
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
        """
        mfa_trigger = optimal_threshold
        # Isolation fires only when fused_risk is materially above the detection
        # boundary — not just barely over it.  A 0.15 offset means a score of
        # 0.89+ (at default threshold 0.7448) triggers full containment; scores
        # between the threshold and 0.89 go to STEP_UP_AUTH unless the threat
        # class is explicitly non-Normal/non-DoS.
        isolation_trigger = min(optimal_threshold + 0.15, 0.90)

        raw_rules = self.playbooks.get("routing_rules", [])

        # Only send the pre-defined native string rules to the LLM.
        # Learned dict rules (LEARNED_NNN / FN_PATCH_NNN) are managed by
        # PlaybookEditorAgent.consolidate_rules() and must not pollute the
        # response-agent prompt with pre-consolidation noise.
        native_rules = [r for r in raw_rules if isinstance(r, str)]

        # Inject the dynamic thresholds into the strings
        formatted_rules = []
        for rule in native_rules:
            formatted_rule = rule.replace("{mfa_trigger}", f"{mfa_trigger:.4f}")
            formatted_rule = formatted_rule.replace("{isolation_trigger}", f"{isolation_trigger:.4f}")
            formatted_rules.append(formatted_rule)
            
        rules_text = "\n        ".join(formatted_rules)

        # ── Slim context — only the fields the rules actually reference ──────
        telemetry = context.get("network_telemetry", {})
        ml = context.get("ml_risk_profile", {})
        slim_context = {
            "fused_risk":   ml.get("fused_risk"),
            "cat_risk":     ml.get("categorical_risk"),
            "threat_class": context.get("predicted_threat_classification"),
            "proto":        telemetry.get("proto"),
            "service":      telemetry.get("service"),
            "state":        telemetry.get("state"),
            "spkts":        telemetry.get("spkts"),
            "dpkts":        telemetry.get("dpkts"),
        }

        system_prompt = (
            "You are a SOC analyst. Apply the rules below in order — first match wins. "
            "Output ONLY a JSON object, no markdown, no preamble\n\n"
            "RULES:\n"
            f"{rules_text}\n\n"
            "EXAMPLES:\n"
            '{"fused_risk":0.85,"threat_class":"Reconnaissance","proto":"udp","state":"INT","spkts":250,"dpkts":0} '
            '-> {"reasoning":"Unidirectional UDP scan, zero response — confirmed recon. Rule 5.","playbook":"NETWORK_ISOLATION","is_false_positive":false}\n'
            '{"fused_risk":0.82,"threat_class":"Normal","proto":"tcp","state":"FIN","spkts":4,"dpkts":4} '
            '-> {"reasoning":"Symmetric TCP teardown, Normal class, low packet count — ML false positive.","playbook":"ALLOW","is_false_positive":true}\n\n'
            "OUTPUT (return exactly this structure, nothing else):\n"
            '{"reasoning":"<1-2 sentences>","playbook":"ALLOW|STEP_UP_AUTH|NETWORK_ISOLATION|RATE_LIMIT_DOS","is_false_positive":true|false}'
        )

        user_prompt = json.dumps(slim_context, cls=_NumpyEncoder)

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "deepseek-r1:8b",
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "options": {
                "temperature": 0.0,
                "num_predict": 3072,
            }
        }

        try:
            response = requests.post(url, json=payload, timeout=240)
            response.raise_for_status()
            raw_text = response.json()["response"]

            # DEBUG — remove once stable; shows first 300 chars so you can see if
            # the model is still overflowing its think chain or outputting garbage
            print(f"[DEBUG] LLM raw ({len(raw_text)} chars): {raw_text[:300]!r}")

            # 1. Strip think tags (even if the model got cut off and forgot the closing tag)
            cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", raw_text, flags=re.DOTALL).strip()
            
            # 2. Aggressively hunt for the JSON object in whatever text is left
            json_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if json_match:
                cleaned = json_match.group(0)
            else:
                raise ValueError("No JSON object found in LLM output.")

            return json.loads(cleaned)
        
        except requests.exceptions.Timeout:
            print(f"[ERROR] LLM Network Timeout: Ollama did not respond within 240 seconds.")
            return {
                "reasoning": "Fail-safe: Ollama network timeout (>240s).",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False,
                "is_llm_failsafe": True
            }

        except ValueError as e:
            # Model hit num_predict cap inside <think> and produced no JSON
            print(f"[ERROR] LLM JSON Parse Failed: {e} — model likely exhausted token budget in <think> chain.")
            return {
                "reasoning": "Fail-safe: model output no parseable JSON (think-chain overflow).",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False,
                "is_llm_failsafe": True
            }

        except Exception as e:
            print(f"[ERROR] LLM Evaluation Failed (unexpected): {type(e).__name__}: {e}")
            return {
                "reasoning": "Fail-safe: unexpected LLM error.",
                "playbook": "NETWORK_ISOLATION",
                "is_false_positive": False,
                "is_llm_failsafe": True
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
        """FIX 2: Uses pre-computed signature from TrustEvaluation.
        FIX (Mutual exclusivity): Never allowlist a signature that is currently
        quarantined — quarantine always wins. The PlaybookEditorAgent's
        _register_quarantine is the only path that can move a sig from quarantine
        to allowlist, and only after explicit FP confirmation."""
        sig = trust_eval.behavior_signature
        if sig in self.active_quarantines:
            print(f"       [MEMORY] '{sig}' is quarantined — allowlist add skipped (quarantine takes precedence).")
            return
        if sig not in self.dynamic_allowlist:
            print(f"       [MEMORY] Adding '{sig}' to Allowlist")
            self.dynamic_allowlist.add(sig)
            self._save_memory()

    def _add_to_quarantine(self, trust_eval: TrustEvaluation):
        """FIX 2: Uses pre-computed signature from TrustEvaluation.
        FIX (Mutual exclusivity): Strips the signature from the dynamic allowlist
        before adding to quarantine so the two sets are always disjoint."""
        sig = trust_eval.behavior_signature
        if sig in self.dynamic_allowlist:
            print(f"       [MEMORY] Removing '{sig}' from Allowlist (confirmed attack — quarantine overrides)")
            self.dynamic_allowlist.discard(sig)
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