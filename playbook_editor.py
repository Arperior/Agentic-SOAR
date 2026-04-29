"""
playbook_editor_agent.py — Autonomous Self-Learning Playbook Editor

Implements the autonomous feedback loop described in the SOAR design.
During training/evaluation, if the SOAR engine makes a mistake 
(e.g., isolates ground-truth NORMAL traffic), it triggers this agent:

  [SOAR makes a False Positive during evaluation]
      ↓
  [PlaybookEditorAgent.autonomous_fp_correction()]
      ↓
  [LLM autonomously audits the context and diagnoses the failure]
      ↓
  [LLM synthesises a new routing rule to prevent recurrence]
      ↓
  [Rule + audit record written to playbooks.json]
      ↓
  [dynamic_allowlist updated in agent memory]
      ↓
  [Future events matching this pattern → ALLOW fast-path]
"""

import json
import datetime
import re
import os
import requests
from typing import Optional
import numpy as np

# ---------------------------------------------------------------------------
# Correction record — persisted to correction_log.json
# ---------------------------------------------------------------------------

class CorrectionRecord:
    """
    Immutable record of an autonomous AI self-correction.
    Stored in correction_log.json for audit and batch review.
    """

    def __init__(
        self,
        event_id,
        context: dict,
        ai_decision: dict,
        new_rule: Optional[dict],
        analysis: str,
        corrected_at: str
    ):
        self.event_id     = event_id
        self.context      = context
        self.ai_decision  = ai_decision
        self.new_rule     = new_rule
        self.analysis     = analysis
        self.corrected_at = corrected_at

    def to_dict(self) -> dict:
        return {
            "event_id":     self.event_id,
            "context":      self.context,
            "ai_decision":  self.ai_decision,
            "new_rule":     self.new_rule,
            "analysis":     self.analysis,
            "corrected_at": self.corrected_at
        }


# ---------------------------------------------------------------------------
# PlaybookEditorAgent
# ---------------------------------------------------------------------------

class PlaybookEditorAgent:
    """
    Autonomous self-learning agent that analyses AI mistakes and writes
    new routing rules into playbooks.json so the same mistake is not
    repeated on future events with a matching signature.

    Parameters
    ----------
    soar_agent : ZeroTrustSOARAgent
        The live SOAR agent whose playbooks and memory this editor manages.
    correction_log_file : str
        Path to the JSON file where correction records are persisted.
    ollama_model : str
        Local LLM model name served by Ollama, used for mistake analysis
        and rule synthesis.
    ollama_url : str
        Base URL for the Ollama HTTP API.
    """

    # Maximum routing rules kept in playbooks.json before compaction
    MAX_ROUTING_RULES = 30

    def __init__(
        self,
        soar_agent,
        correction_log_file: str = "correction_log.json",
        ollama_model: str = "deepseek-r1:8b",
        ollama_url: str = "http://localhost:11434/api/generate"
    ):
        self.soar_agent          = soar_agent
        self.correction_log_file = correction_log_file
        self.ollama_model        = ollama_model
        self.ollama_url          = ollama_url

        self.correction_log: list[dict] = self._load_correction_log()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def autonomous_fp_correction(
        self,
        event_id,
        context: dict,
        decision: dict
    ) -> CorrectionRecord:
        """
        Main entry point — triggered autonomously during evaluation when
        ground-truth Normal traffic is wrongly escalated by the AI.

        Steps
        -----
        1. Extract the behaviour signature from the context.
        2. Ask the LLM to autonomously diagnose WHY the AI made the wrong decision.
        3. Ask the LLM to synthesise a new routing rule that prevents it.
        4. Append the rule to playbooks.json (routing_rules section).
        5. Add the signature to the dynamic allowlist in agent memory.
        6. Persist the correction record to correction_log.json.

        Parameters
        ----------
        event_id    : int | str
        context     : dict  — original context passed to soar_agent.run()
        decision    : dict  — what the AI returned (playbook, reasoning, …)

        Returns
        -------
        CorrectionRecord
        """
        event_id = int(event_id) if hasattr(event_id, "item") else event_id

        print(f"\n{'='*60}")
        print(f"[PlaybookEditorAgent] Triggering Autonomous Self-Healing for Event {event_id}")
        print(f"  AI decided  : {decision.get('playbook')} — {decision.get('reasoning', '')[:80]}")
        print(f"  Ground Truth: NORMAL (False Positive Detected)")

        # ── Step 1: Extract the behaviour signature ───────────────────
        telemetry  = context.get("network_telemetry", {})
        trust_eval_dict = decision.get("trust_eval") or {}
        sig = trust_eval_dict.get("behavior_signature") or self._derive_signature(telemetry)

        print(f"  Behaviour sig: {sig}")

        # ── Step 2: Diagnose the mistake ──────────────────────────────
        analysis = self._analyse_mistake(context, decision)
        print(f"\n[PlaybookEditorAgent] Autonomous Root-cause analysis:\n  {analysis}")

        # ── Step 3: Synthesise a new routing rule ─────────────────────
        new_rule = self._synthesise_rule(context, decision, analysis)
        print(f"\n[PlaybookEditorAgent] Synthesised patch rule:\n  {json.dumps(new_rule, indent=4)}")

        # ── Step 4: Write the rule to playbooks.json ──────────────────
        if new_rule:
            self._append_rule_to_playbooks(new_rule, event_id)

        # ── Step 5: Add signature to dynamic allowlist ────────────────
        self._register_allowlist(sig)

        # ── Step 6: Persist correction record ─────────────────────────
        record = CorrectionRecord(
            event_id     = event_id,
            context      = context,
            ai_decision  = decision,
            new_rule     = new_rule,
            analysis     = analysis,
            corrected_at = datetime.datetime.now().isoformat()
        )
        self.correction_log.append(record.to_dict())
        self._save_correction_log()

        print(f"\n[PlaybookEditorAgent] ✓ Self-Healing complete. "
              f"Playbooks updated. Signature '{sig}' added to allowlist.")
        print(f"{'='*60}\n")

        return record

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    def _analyse_mistake(
        self,
        context: dict,
        decision: dict
    ) -> str:
        """
        Asks the local LLM to act as an internal auditor and diagnose
        why the SOAR logic wrongly isolated this ground-truth benign event.
        """
        telemetry   = context.get("network_telemetry", {})
        ml_profile  = context.get("ml_risk_profile", {})
        trust_factors = (decision.get("trust_eval") or {}).get("factors", [])

        system_prompt = """
You are the internal Cognitive Auditor for an AI-driven SOAR system.
The system is currently undergoing evaluation, and you have detected a False Positive:
The SOAR engine escalated and contained an event that is actually GROUND-TRUTH NORMAL benign traffic.

Your task: Explain in 2-4 concise technical sentences WHY the AI made the wrong decision. 
Consider:
  - Which ML signal (fused_risk, categorical_risk, anomaly_risk) was artificially high?
  - Which policy penalties (trust factors) pushed the trust score below threshold?
  - What specific feature of the traffic (proto, service, state, spkts/dpkts) confused the models?
  - Was this a known model hallucination (like CatBoost misclassifying UDP traffic as Shellcode)?

Output ONLY plain text — no JSON, no markdown, no preamble.
"""

        user_prompt = (
            f"AI decision   : {decision.get('playbook')}\n"
            f"AI reasoning  : {decision.get('reasoning', 'none')}\n"
            f"ML profile    : {json.dumps(ml_profile)}\n"
            f"Trust factors : {json.dumps(trust_factors)}\n"
            f"Telemetry     : {json.dumps({k: telemetry.get(k) for k in ['proto','service','state','spkts','dpkts']})}\n"
            f"Threat class  : {context.get('predicted_threat_classification')}\n\n"
            f"Diagnose the root cause of this autonomous False Positive."
        )

        return self._call_llm(system_prompt, user_prompt, max_tokens=1500) or (
            "Unable to generate analysis: LLM unavailable. Autonomous fallback triggered."
        )

    def _synthesise_rule(
        self,
        context: dict,
        decision: dict,
        analysis: str
    ) -> Optional[dict]:
        """
        Asks the local LLM to write a new routing rule that would have
        correctly classified this event.
        """
        telemetry   = context.get("network_telemetry", {})
        ml_profile  = context.get("ml_risk_profile", {})
        threat_type = context.get("predicted_threat_classification", "UNKNOWN")

        existing_rules = self.soar_agent.playbooks.get("routing_rules", [])
        learned_count  = sum(
            1 for r in existing_rules
            if isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent"
        )
        rule_id = f"LEARNED_{learned_count + 1:03d}"

        system_prompt = """
You are the Cognitive Auditor writing a self-healing patch for a SOAR playbook. 
A False Positive occurred, and your job is to write ONE precise routing rule that 
will correctly classify similar benign events as 'ALLOW' in the future.

The rule must be specific enough to avoid over-broad exceptions that would allow real attacks through. 
Anchor it to ≥2 observable signal features (e.g. proto + state + dpkts, or threat_class + fused_risk_range).

Output ONLY a raw JSON object with these exact keys:
{
  "id":        "<LEARNED_NNN>",
  "condition": "IF <specific observable conditions> ...",
  "playbook":  "ALLOW",
  "is_fp":     true,
  "rationale": "<1-2 sentences explaining why this patches the ML hallucination>"
}

No markdown, no preamble, no extra keys.
"""

        user_prompt = (
            f"Rule ID to assign : {rule_id}\n"
            f"Root-cause analysis:\n{analysis}\n\n"
            f"ML profile        : {json.dumps(ml_profile)}\n"
            f"Threat class      : {threat_type}\n"
            f"Key telemetry     : proto={telemetry.get('proto')}, service={telemetry.get('service')}, "
            f"state={telemetry.get('state')}, spkts={telemetry.get('spkts')}, dpkts={telemetry.get('dpkts')}\n\n"
            f"Write the JSON patch rule."
        )

        raw = self._call_llm(system_prompt, user_prompt, max_tokens=1500)
        if not raw:
            return None

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            for fence in ("```json", "```"):
                if cleaned.startswith(fence):
                    cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            rule = json.loads(cleaned)
            rule["created_at"] = datetime.datetime.now().isoformat()
            rule["source"]     = "PlaybookEditorAgent"
            return rule

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[PlaybookEditorAgent] Rule synthesis parse error: {exc}")
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 400) -> Optional[str]:
        payload = {
            "model":   self.ollama_model,
            "system":  system_prompt,
            "prompt":  user_prompt,
            "stream":  False,
            "options": {
                "temperature": 0.1,
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=240)
            response.raise_for_status()
            raw = response.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except requests.exceptions.Timeout:
            print("[PlaybookEditorAgent] LLM timeout during analysis.")
            return None
        except Exception as exc:
            print(f"[PlaybookEditorAgent] LLM call failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Playbook I/O
    # ------------------------------------------------------------------

    def _append_rule_to_playbooks(self, new_rule: dict, event_id):
        playbooks = self.soar_agent.playbooks

        if "routing_rules" not in playbooks:
            playbooks["routing_rules"] = []
        if "learned_rule_audit" not in playbooks:
            playbooks["learned_rule_audit"] = []

        routing_rules = playbooks["routing_rules"]

        existing_conditions = set()
        for r in routing_rules:
            if isinstance(r, dict):
                existing_conditions.add(r.get("condition", "").strip().lower())

        if new_rule.get("condition", "").strip().lower() in existing_conditions:
            print("[PlaybookEditorAgent] Rule with identical condition already exists — skipping insert.")
            return

        if len(routing_rules) >= self.MAX_ROUTING_RULES:
            self._compact_rules(playbooks)

        # Insert at the TOP of the routing rules (index 0) so the patch overrides older logic
        routing_rules.insert(0, new_rule)

        playbooks["learned_rule_audit"].append({
            "rule_id":      new_rule["id"],
            "event_id":     event_id,
            "trigger":      "Autonomous_Evaluation_FP",
            "appended_at":  new_rule["created_at"]
        })

        self._save_playbooks(playbooks)
        print(f"[PlaybookEditorAgent] Rule '{new_rule['id']}' written to {self.soar_agent.playbook_file}")

    def _compact_rules(self, playbooks: dict):
        routing_rules = playbooks["routing_rules"]
        learned = [r for r in routing_rules if isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent"]
        native  = [r for r in routing_rules if not (isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent")]

        headroom = 5
        keep_count = max(0, self.MAX_ROUTING_RULES - len(native) - headroom)
        learned_trimmed = learned[-keep_count:] if keep_count else []

        playbooks["routing_rules"] = learned_trimmed + native
        print(f"[PlaybookEditorAgent] Compacted routing_rules: kept {len(learned_trimmed)} learned rules.")

    def _save_playbooks(self, playbooks: dict):
        with open(self.soar_agent.playbook_file, "w") as f:
            json.dump(playbooks, f, indent=4)
        self.soar_agent.playbooks = playbooks

    # ------------------------------------------------------------------
    # Memory helpers
    # ------------------------------------------------------------------

    def _register_allowlist(self, sig: str):
        if sig not in self.soar_agent.dynamic_allowlist:
            self.soar_agent.dynamic_allowlist.add(sig)
            self.soar_agent.active_quarantines.discard(sig)
            self.soar_agent._save_memory()
            print(f"[PlaybookEditorAgent] '{sig}' added to dynamic_allowlist (removed from quarantine if present).")

    @staticmethod
    def _derive_signature(telemetry: dict) -> str:
        proto   = telemetry.get("proto",   "unknown")
        service = telemetry.get("service", "unknown")
        state   = telemetry.get("state",   "unknown")
        return f"{proto}_{service}_{state}"

    # ------------------------------------------------------------------
    # Correction log I/O
    # ------------------------------------------------------------------

    def _load_correction_log(self) -> list:
        if not os.path.exists(self.correction_log_file):
            return []
        with open(self.correction_log_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return []

    def _save_correction_log(self):

        def _scrub_numpy(obj):
            if isinstance(obj, dict):
                return {k: _scrub_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_scrub_numpy(v) for v in obj]
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if hasattr(obj, "item"):  # catches pandas + numpy edge cases
                try:
                    return obj.item()
                except:
                    pass
            return obj

        safe_log = _scrub_numpy(self.correction_log)

        with open(self.correction_log_file, "w") as f:
            json.dump(safe_log, f, indent=4)