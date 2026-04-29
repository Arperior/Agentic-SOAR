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

    def autonomous_fn_correction(
        self,
        event_id,
        context: dict,
        decision: dict
    ) -> CorrectionRecord:
        """
        Triggered when ground-truth ATTACK traffic was wrongly ALLOWed (False Negative).
        Higher severity than FP — a real threat was missed.

        Steps
        -----
        1. Extract the behaviour signature.
        2. Ask the LLM to diagnose WHY the threat was missed.
        3. Synthesise a NETWORK_ISOLATION rule — only if LLM analysis succeeded.
        4. Append the rule to playbooks.json at the TOP (highest priority).
        5. Remove the signature from dynamic_allowlist and add to quarantine.
        6. Persist the correction record to correction_log.json.
        """
        event_id = int(event_id) if hasattr(event_id, "item") else event_id

        # Do not write rules or update quarantine when the isolation decision came
        # from the LLM fail-safe (empty/timeout response).  A fail-safe fires
        # because Ollama was unreachable — it carries zero diagnostic signal and
        # must never be used to anchor a patch rule or permanently quarantine a
        # signature.  The event will be re-evaluated on the next run once the LLM
        # recovers.
        if decision.get("is_llm_failsafe"):
            print(f"\n[PlaybookEditorAgent] \u26a0 Skipping FN self-healing for Event {event_id} "
                  f"\u2014 isolation was triggered by LLM fail-safe, not a genuine model decision.")
            return None


        print(f"\n{'='*60}")
        print(f"[PlaybookEditorAgent] \u26a0 FALSE NEGATIVE \u2014 Missed Attack for Event {event_id}")
        print(f"  AI decided  : {decision.get('playbook')} \u2014 {decision.get('reasoning', '')[:80]}")
        print(f"  Ground Truth: ATTACK (False Negative \u2014 threat was missed!)")

        telemetry       = context.get("network_telemetry", {})
        trust_eval_dict = decision.get("trust_eval") or {}
        sig = trust_eval_dict.get("behavior_signature") or self._derive_signature(telemetry)

        print(f"  Behaviour sig: {sig}")

        # Step 2: Diagnose
        analysis = self._analyse_fn_mistake(context, decision)
        print(f"\n[PlaybookEditorAgent] FN Root-cause analysis:\n  {analysis}")

        # Step 3: Only synthesise if LLM produced a real analysis
        if "LLM unavailable" in analysis:
            print("\n[PlaybookEditorAgent] Skipping FN rule synthesis — no valid analysis to anchor the rule.")
            new_rule = None
        else:
            new_rule = self._synthesise_fn_rule(context, decision, analysis)
        print(f"\n[PlaybookEditorAgent] Synthesised FN patch rule:\n  {json.dumps(new_rule, indent=4)}")

        # Step 4: Write rule
        if new_rule:
            self._append_rule_to_playbooks(new_rule, event_id, trigger="Autonomous_Evaluation_FN")

        # Step 5: Move sig to quarantine, remove from allowlist
        self._register_quarantine(sig)

        clean_context  = json.loads(json.dumps(context,  default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
        clean_decision = json.loads(json.dumps(decision, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

        # Step 6: Persist
        record = CorrectionRecord(
            event_id     = event_id,
            context      = clean_context,
            ai_decision  = clean_decision,
            new_rule     = new_rule,
            analysis     = analysis,
            corrected_at = datetime.datetime.now().isoformat()
        )
        self.correction_log.append(record.to_dict())
        self._save_correction_log()

        print(f"\n[PlaybookEditorAgent] \u2713 FN Self-Healing complete. Signature '{sig}' moved to quarantine.")
        print(f"{'='*60}\n")

        return record

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

        # Fail-safe decisions (LLM timeout / empty response) carry no diagnostic
        # signal — the isolation was a precautionary default, not a model judgement.
        # Writing an ALLOW rule or allowlisting the signature on the basis of a
        # fail-safe would permanently exempt that signature from scrutiny on
        # grounds of a network error, not actual benign behaviour.
        if decision.get("is_llm_failsafe"):
            print(f"\n[PlaybookEditorAgent] \u26a0 Skipping FP self-healing for Event {event_id} "
                  f"\u2014 NETWORK_ISOLATION was triggered by LLM fail-safe, not a genuine model decision. "
                  f"No rule written, no allowlist update.")
            return None


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

        # ── Step 3: Synthesise a new routing rule (only if LLM produced real analysis)
        if "LLM unavailable" in analysis:
            print("\n[PlaybookEditorAgent] Skipping rule synthesis — no valid analysis to anchor the rule.")
            new_rule = None
        else:
            new_rule = self._synthesise_rule(context, decision, analysis)
        print(f"\n[PlaybookEditorAgent] Synthesised patch rule:\n  {json.dumps(new_rule, indent=4)}")

        # ── Step 4: Write the rule to playbooks.json ──────────────────
        if new_rule:
            self._append_rule_to_playbooks(new_rule, event_id)

        # ── Step 5: Add signature to dynamic allowlist ────────────────
        self._register_allowlist(sig)

        clean_context = json.loads(json.dumps(context, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))
        clean_decision = json.loads(json.dumps(decision, default=lambda o: o.item() if hasattr(o, 'item') else str(o)))

        record = CorrectionRecord(
            event_id     = event_id,
            context      = clean_context,
            ai_decision  = clean_decision,   # ← use sanitised version
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


    def _analyse_fn_mistake(self, context: dict, decision: dict) -> str:
        """Asks the LLM to diagnose WHY a real attack was missed (ALLOWed)."""
        telemetry     = context.get("network_telemetry", {})
        ml_profile    = context.get("ml_risk_profile", {})
        trust_factors = (decision.get("trust_eval") or {}).get("factors", [])

        system_prompt = """
You are the internal Cognitive Auditor for an AI-driven SOAR system.
A FALSE NEGATIVE was detected: the SOAR engine ALLOWed traffic that is actually
a GROUND-TRUTH ATTACK. This is a critical security failure.

Your task: Explain in 2-4 concise technical sentences WHY the threat was missed.
Consider:
  - Was the fused_risk or anomaly_risk too low despite real attack activity?
  - Did a trust bonus (allowlist, high trust score) override a risk signal?
  - Did the threat classification cause a wrong fast-path?
  - What feature of the traffic (proto, service, state, spkts/dpkts) disguised the attack?

Output ONLY plain text — no JSON, no markdown, no preamble.
"""
        user_prompt = (
            f"AI decision   : {decision.get('playbook')}\n"
            f"AI reasoning  : {decision.get('reasoning', 'none')}\n"
            f"ML profile    : {json.dumps(ml_profile)}\n"
            f"Trust factors : {json.dumps(trust_factors)}\n"
            f"Telemetry     : {json.dumps({k: telemetry.get(k) for k in ['proto','service','state','spkts','dpkts']})}\n"
            f"Threat class  : {context.get('predicted_threat_classification')}\n\n"
            f"Diagnose the root cause of this False Negative (missed attack)."
        )

        return self._call_llm(system_prompt, user_prompt, max_tokens=1500) or (
            "Unable to generate FN analysis: LLM unavailable. Autonomous fallback triggered."
        )

    def _synthesise_fn_rule(self, context: dict, decision: dict, analysis: str) -> Optional[dict]:
        """
        Synthesises a NETWORK_ISOLATION rule to catch similar missed attacks in future.
        Returns None if the LLM is unavailable or output is unrecoverable — no static fallback.
        """
        telemetry   = context.get("network_telemetry", {})
        ml_profile  = context.get("ml_risk_profile", {})
        threat_type = context.get("predicted_threat_classification", "UNKNOWN")

        existing_rules = self.soar_agent.playbooks.get("routing_rules", [])
        learned_count  = sum(
            1 for r in existing_rules
            if isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent"
        )
        rule_id = f"FN_PATCH_{learned_count + 1:03d}"

        system_prompt = """
You are a SOAR playbook auditor. A False Negative occurred — a real attack was ALLOWed.
Write ONE precise JSON routing rule to catch similar attacks as NETWORK_ISOLATION in future.

RULES:
- Anchor condition to 2+ signal features from the telemetry provided.
- condition format: plain English with AND only, e.g. "proto=tcp AND state=FIN AND threat_class=Fuzzers"
- Do NOT use == operators, quotes, or backslashes inside the condition string.
- rationale: max 20 words. Be direct.
- Respond with ONLY the JSON object. No explanation, no markdown, no preamble.

{
  "id":        "<FN_PATCH_NNN>",
  "condition": "proto=X AND state=Y AND threat_class=Z",
  "playbook":  "NETWORK_ISOLATION",
  "is_fn":     true,
  "rationale": "One concise sentence max 20 words."
}
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

        raw = self._call_llm(system_prompt, user_prompt, max_tokens=3000)
        if not raw:
            print("[PlaybookEditorAgent] LLM unavailable — skipping FN rule synthesis. No rule written.")
            return None

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            for fence in ("```json", "```"):
                if cleaned.startswith(fence):
                    cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            if not cleaned.endswith("}"):
                if cleaned.count('"') % 2:
                    cleaned += '"'
                depth = cleaned.count("{") - cleaned.count("}")
                cleaned += "}" * max(depth, 1)
                print("[PlaybookEditorAgent] Truncated FN JSON — attempted structural repair.")

            rule = json.loads(cleaned)
            rule["condition"]  = PlaybookEditorAgent._normalise_condition(rule.get("condition", ""))
            rule["created_at"] = datetime.datetime.now().isoformat()
            rule["source"]     = "PlaybookEditorAgent"
            return rule

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[PlaybookEditorAgent] FN rule parse error after repair attempt: {exc}")
            print("[PlaybookEditorAgent] No FN rule written — LLM output was unrecoverable.")
            return None

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
You are a SOAR playbook auditor. A False Positive occurred — benign traffic was wrongly isolated.
Write ONE precise JSON routing rule to correctly ALLOW similar benign events in future.

RULES:
- Anchor condition to 2+ signal features from the telemetry provided.
- condition format: plain English with AND only, e.g. "proto=tcp AND state=FIN AND threat_class=Fuzzers"
- Do NOT use == operators, quotes, or backslashes inside the condition string.
- rationale: max 20 words. Be direct.
- Respond with ONLY the JSON object. No explanation, no markdown, no preamble.

{
  "id":        "<LEARNED_NNN>",
  "condition": "proto=X AND state=Y AND threat_class=Z",
  "playbook":  "ALLOW",
  "is_fp":     true,
  "rationale": "One concise sentence max 20 words."
}
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

        raw = self._call_llm(system_prompt, user_prompt, max_tokens=3000)
        if not raw:
            print("[PlaybookEditorAgent] LLM unavailable — skipping rule synthesis. No rule written.")
            return None

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            for fence in ("```json", "```"):
                if cleaned.startswith(fence):
                    cleaned = cleaned[len(fence):]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            cleaned = cleaned.strip()

            # Repair truncated JSON caused by hitting num_predict token limit
            if not cleaned.endswith("}"):
                if cleaned.count('"') % 2:
                    cleaned += '"'
                depth = cleaned.count("{") - cleaned.count("}")
                cleaned += "}" * max(depth, 1)
                print("[PlaybookEditorAgent] Truncated JSON detected — attempted structural repair.")

            rule = json.loads(cleaned)
            rule["condition"]  = PlaybookEditorAgent._normalise_condition(rule.get("condition", ""))
            rule["created_at"] = datetime.datetime.now().isoformat()
            rule["source"]     = "PlaybookEditorAgent"
            return rule

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[PlaybookEditorAgent] Rule synthesis parse error after repair attempt: {exc}")
            print("[PlaybookEditorAgent] No rule written — LLM output was unrecoverable.")
            return None

    def _call_llm(self, system_prompt: str, user_prompt: str, max_tokens: int = 2000) -> Optional[str]:
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

    @staticmethod
    def _normalise_condition(condition: str) -> str:
        """
        Cleans LLM-generated condition strings into a consistent readable format.
        Strips leading IF, == operators, backslashes, and quotes around values.
        e.g. 'IF proto == \"tcp\" AND state == \"FIN\"' -> 'proto=tcp AND state=FIN'
        """
        c = condition
        c = re.sub(r"^IF\s+", "", c, flags=re.IGNORECASE).strip()
        c = c.replace("==", "=")
        c = c.replace('\\"', "").replace("\'", "").replace('"', "").replace("'", "")
        c = re.sub(r"\s{2,}", " ", c).strip()
        return c

    def _append_rule_to_playbooks(self, new_rule: dict, event_id, trigger: str = "Autonomous_Evaluation_FP"):
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
            "trigger":      trigger,
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


    def _register_quarantine(self, sig: str):
        """
        Moves a signature to active_quarantines and strips it from the allowlist.
        Called after a confirmed False Negative to harden future routing.
        """
        if sig in self.soar_agent.dynamic_allowlist:
            self.soar_agent.dynamic_allowlist.discard(sig)
            print(f"[PlaybookEditorAgent] \u26a0 '{sig}' REMOVED from dynamic_allowlist (confirmed missed attack).")
        if sig not in self.soar_agent.active_quarantines:
            self.soar_agent.active_quarantines.add(sig)
            self.soar_agent._save_memory()
            print(f"[PlaybookEditorAgent] '{sig}' added to active_quarantines.")

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
            # Explicitly cover np.int64, np.int32, etc.
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            # Fallback for any other numpy scalar
            if hasattr(obj, "item"):
                try:
                    return obj.item()
                except Exception:
                    pass
            return obj

        safe_log = _scrub_numpy(self.correction_log)

        with open(self.correction_log_file, "w") as f:
            json.dump(safe_log, f, indent=4)  # ← make sure this is safe_log