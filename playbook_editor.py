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

        # Rule health tracking — records how many FNs and FPs each rule
        # (native or learned) has been implicated in. Rules that exceed
        # FN_THRESHOLD are flagged for human review.
        self.rule_health_file = "rule_health.json"
        self.rule_health: dict = self._load_rule_health()

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
        sig = trust_eval_dict.get("behavior_signature") or self._derive_signature(
            telemetry, threat_class=context.get("predicted_threat_classification", "unknown")
        )

        print(f"  Behaviour sig: {sig}")

        # Step 2: Diagnose
        analysis = self._analyse_fn_mistake(context, decision)
        print(f"\n[PlaybookEditorAgent] FN Root-cause analysis:\n  {analysis}")

        # Step 2b: Attribute the FN to whichever rule the LLM cited in its reasoning.
        # This feeds the rule health tracker so persistently bad rules surface for review.
        implicated = self._extract_implicated_rule(
            (decision.get("reasoning") or "") + " " + analysis
        )
        if implicated:
            print(f"[PlaybookEditorAgent] Rule implicated in FN: '{implicated}'")
            self.record_rule_implicated_in_fn(implicated, event_id, decision.get("reasoning", ""))

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
        sig = trust_eval_dict.get("behavior_signature") or self._derive_signature(
            telemetry, threat_class=context.get("predicted_threat_classification", "unknown")
        )

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
    # Rule health tracking — public API
    # ------------------------------------------------------------------

    def record_rule_implicated_in_fn(self, rule_id: str, event_id, reasoning: str):
        """
        Increments the FN counter for rule_id and flags the rule for human
        review once it crosses FN_THRESHOLD. Call after every confirmed FN
        where the root-cause analysis names a specific rule.

        FN_THRESHOLD is intentionally low (2) so problems surface quickly
        during a 50-event evaluation run. Raise it for production use.
        """
        FN_THRESHOLD = 2

        if rule_id not in self.rule_health:
            self.rule_health[rule_id] = {
                "fn_count": 0, "fp_count": 0,
                "flagged": False, "incidents": []
            }

        self.rule_health[rule_id]["fn_count"] += 1
        self.rule_health[rule_id]["incidents"].append({
            "type": "FN",
            "event_id": event_id,
            "reasoning_snippet": reasoning[:120],
            "recorded_at": datetime.datetime.now().isoformat()
        })

        fn_count = self.rule_health[rule_id]["fn_count"]
        if fn_count >= FN_THRESHOLD and not self.rule_health[rule_id]["flagged"]:
            self.rule_health[rule_id]["flagged"] = True
            print(f"\n[PlaybookEditorAgent] ⚠ RULE HEALTH ALERT: Rule '{rule_id}' has caused "
                  f"{fn_count} false negatives — flagged for human review.")
            print(f"  → Run review_flagged_rules() or check rule_health.json for full history.")

        self._save_rule_health()

    def review_flagged_rules(self):
        """
        Prints a summary of all rules flagged due to repeated FNs.
        Call at the end of an evaluation run.

        For native rules (NATIVE_RULE_N): prints the rule text so a human
        can decide whether to tighten, narrow, or remove it from
        _get_default_playbooks() in soar_zta.py.

        For learned rules (LEARNED_NNN / FN_PATCH_NNN): prints the condition
        and playbook so it can be edited in playbooks.json or removed and
        re-synthesised by the next FN correction cycle.
        """
        flagged = {
            rid: data for rid, data in self.rule_health.items()
            if data.get("flagged")
        }

        if not flagged:
            print("[PlaybookEditorAgent] Rule health: all rules within acceptable FN tolerance.")
            return

        routing_rules = self.soar_agent.playbooks.get("routing_rules", [])

        print(f"\n{'='*60}")
        print(f"[PlaybookEditorAgent] RULE HEALTH REPORT — {len(flagged)} rule(s) flagged for review")

        for rule_id, data in flagged.items():
            print(f"\n  ── Rule: {rule_id}")
            print(f"     FN count : {data['fn_count']}  |  FP count: {data['fp_count']}")

            # Locate and print the rule definition
            if rule_id.startswith("NATIVE_RULE_"):
                idx = int(rule_id.split("_")[-1]) - 1
                native_rules = [r for r in routing_rules if isinstance(r, str)]
                if 0 <= idx < len(native_rules):
                    print(f"     Definition : {native_rules[idx][:140]}")
                    print(f"     Action     : Edit _get_default_playbooks() in soar_zta.py — "
                          f"tighten condition or remove rule.")
            else:
                matched = next(
                    (r for r in routing_rules
                     if isinstance(r, dict) and r.get("id") == rule_id),
                    None
                )
                if matched:
                    print(f"     Condition  : {matched.get('condition')}")
                    print(f"     Playbook   : {matched.get('playbook')}")
                    print(f"     Rationale  : {matched.get('rationale', 'n/a')}")
                    print(f"     Action     : Edit or delete from playbooks.json, "
                          f"then call consolidate_rules().")
                else:
                    print(f"     Definition : <rule no longer in playbooks.json — already replaced or compacted>")

            print(f"     Recent incidents:")
            for inc in data["incidents"][-3:]:
                print(f"       [{inc['type']}] Event {inc['event_id']}: {inc['reasoning_snippet'][:100]}")

        print(f"\n  Native rules require manual edits. Learned rules can be removed from")
        print(f"  playbooks.json — the next FN correction cycle will re-synthesise a better one.")
        print(f"{'='*60}\n")

    @staticmethod
    def _extract_implicated_rule(text: str) -> Optional[str]:
        """
        Parses LLM reasoning / analysis text to find a rule reference.

        Matches:
          - Learned / patch rule IDs:  LEARNED_003, FN_PATCH_002
          - Native rule references:    "rule 4", "Rule 4", "rule #4"

        Returns a canonical rule ID string, or None if nothing is found.
        """
        # Learned / patch rule IDs take priority (they're unambiguous)
        match = re.search(r"\b(LEARNED_\d+|FN_PATCH_\d+)\b", text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

        # Native rule references like "rule 4", "Rule 4", "rule #4"
        match = re.search(r"\brule\s+#?(\d+)\b", text, re.IGNORECASE)
        if match:
            return f"NATIVE_RULE_{match.group(1)}"

        return None

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
                # FIX (LLM timeout): Honour the caller-supplied max_tokens as a
                # hard cap on token generation. Analysis calls pass 1500, rule
                # synthesis passes 3000 — both well above what the output needs
                # but low enough to prevent runaway <think> chains from timing out.
                "num_predict": max_tokens
            }
        }

        try:
            response = requests.post(self.ollama_url, json=payload, timeout=240)
            response.raise_for_status()
            raw = response.json().get("response", "")
            return re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
        except requests.exceptions.Timeout:
            print(f"[ERROR] LLM Evaluation Timeout: Ollama did not respond within 240 seconds.")
            return {"reasoning": "Fail-safe: network timeout.", "playbook": "NETWORK_ISOLATION",
                    "is_false_positive": False, "is_llm_failsafe": True}

        except ValueError as e:
            # JSON parse failure — model generated think chain but no valid JSON
            print(f"[ERROR] LLM JSON Parse Failed: {e}")
            return {"reasoning": "Fail-safe: model output no parseable JSON (likely think-chain overflow).",
                    "playbook": "NETWORK_ISOLATION", "is_false_positive": False, "is_llm_failsafe": True}

        except Exception as e:
            print(f"[ERROR] LLM Evaluation Failed (unexpected): {e}")
            return {"reasoning": "Fail-safe: unexpected LLM error.", "playbook": "NETWORK_ISOLATION",
                    "is_false_positive": False, "is_llm_failsafe": True}

    # ------------------------------------------------------------------
    # Playbook I/O
    # ------------------------------------------------------------------

    @staticmethod
    def _conditions_overlap(cond_a: str, cond_b: str) -> bool:
        """
        Returns True if one condition's clauses are a subset of the other's.

        e.g. "proto=udp AND state=INT" overlaps (and is broader than)
             "proto=udp AND state=INT AND threat_class=Shellcode"

        Used to prevent near-duplicate rule accumulation: if an existing rule
        already covers the new pattern more broadly, the insert is skipped.
        If the new rule is broader, the old one is replaced.
        """
        def parse_clauses(c: str) -> set:
            return {clause.strip().lower() for clause in c.split("AND") if clause.strip()}

        a = parse_clauses(cond_a)
        b = parse_clauses(cond_b)
        # One is a subset of the other → they cover overlapping traffic
        return a.issubset(b) or b.issubset(a)

    def consolidate_rules(self):
        """
        Asks the LLM to review all learned routing rules and merge near-duplicates
        into broader, more general rules. Call periodically (e.g. every 10 events)
        or at the end of an evaluation run to prevent rule bloat.

        FIX (Rule specificity / bloat): The LLM tends to anchor synthesised rules
        on exact telemetry values (spkts, dpkts) producing hyper-specific rules
        that cover only a single event. Periodic consolidation merges these into
        broader pattern rules (e.g. proto=udp AND state=INT AND threat_class=Shellcode)
        that generalise across the signature class.

        Native string rules (the default routing_rules defined in _get_default_playbooks)
        are never touched — only PlaybookEditorAgent-sourced dicts are consolidated.
        """
        playbooks = self.soar_agent.playbooks
        learned = [
            r for r in playbooks.get("routing_rules", [])
            if isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent"
        ]

        if len(learned) < 3:
            return  # Nothing worth consolidating yet

        print(f"\n[PlaybookEditorAgent] Consolidating {len(learned)} learned rules…")

        system_prompt = """
You are a SOAR playbook auditor reviewing a set of autonomously learned routing rules.
Your task: identify rules that are redundant, near-duplicate, or overly specific and merge
them into broader rules that generalise across the pattern class.

MERGE CRITERIA:
- Two rules whose conditions share the same proto, state, and threat_class (differing only
  in packet counts or minor telemetry values) should be merged into one rule that drops
  the overly-specific clauses.
- Rules that differ in playbook (ALLOW vs NETWORK_ISOLATION) must NEVER be merged.
- Keep rules that are genuinely distinct (different proto, state, or threat_class).

OUTPUT FORMAT:
- Return ONLY a raw JSON array of the consolidated rules.
- Preserve the earliest created_at of any merged group.
- Set "source" to "PlaybookEditorAgent" on every rule.
- Normalise condition strings: plain AND-separated key=value pairs, no quotes, no operators.
- No markdown, no preamble, no explanation — ONLY the JSON array.
"""

        user_prompt = (
            f"Learned rules to consolidate:\n{json.dumps(learned, indent=2)}\n\n"
            f"Return the consolidated JSON array. Merge near-duplicates. Keep distinct rules as-is."
        )

        raw = self._call_llm(system_prompt, user_prompt, max_tokens=3000)
        if not raw:
            print("[PlaybookEditorAgent] Consolidation skipped — LLM unavailable.")
            return

        try:
            cleaned = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL).strip()
            cleaned = re.sub(r"^```json\s*|^```\s*|```\s*$", "", cleaned, flags=re.MULTILINE).strip()

            consolidated = json.loads(cleaned)
            if not isinstance(consolidated, list):
                print("[PlaybookEditorAgent] Consolidation parse error — LLM did not return a list. Skipping.")
                return

            # Normalise all condition strings
            for r in consolidated:
                if isinstance(r, dict):
                    r["condition"] = self._normalise_condition(r.get("condition", ""))

            # Rebuild routing_rules: consolidated learned rules at front, native rules preserved
            native = [
                r for r in playbooks["routing_rules"]
                if not (isinstance(r, dict) and r.get("source") == "PlaybookEditorAgent")
            ]
            playbooks["routing_rules"] = consolidated + native
            self._save_playbooks(playbooks)
            print(f"[PlaybookEditorAgent] Consolidation complete: {len(learned)} rules → {len(consolidated)} rules.")

        except (json.JSONDecodeError, ValueError) as exc:
            print(f"[PlaybookEditorAgent] Consolidation parse error: {exc} — existing rules unchanged.")

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
        new_cond = new_rule.get("condition", "").strip().lower()
        new_playbook = new_rule.get("playbook", "")

        # FIX (Rule bloat / near-duplicate accumulation):
        # The original check used exact-string matching only, so rules differing
        # by a single telemetry clause (e.g. adding spkts=2) were both written.
        # The new logic:
        #   1. Exact match → skip (unchanged behaviour).
        #   2. Overlapping condition AND same playbook direction:
        #      - If existing rule is broader (fewer clauses) → skip: already covered.
        #      - If new rule is broader (fewer clauses) → replace old with new rule.
        #   3. Overlapping condition but DIFFERENT playbook (ALLOW vs NETWORK_ISOLATION)
        #      → always insert: conflicting rules need both to be visible for review.
        rule_to_replace = None
        for r in routing_rules:
            if not isinstance(r, dict):
                continue
            existing_cond = r.get("condition", "").strip().lower()
            existing_playbook = r.get("playbook", "")

            if existing_cond == new_cond:
                print(f"[PlaybookEditorAgent] Identical rule '{r.get('id')}' already exists — skipping insert.")
                return

            if self._conditions_overlap(existing_cond, new_cond) and existing_playbook == new_playbook:
                existing_clauses = len([c for c in existing_cond.split("AND") if c.strip()])
                new_clauses      = len([c for c in new_cond.split("AND")      if c.strip()])

                if existing_clauses <= new_clauses:
                    # Existing rule is equally broad or broader → new rule is redundant
                    print(f"[PlaybookEditorAgent] Broader rule '{r.get('id')}' already covers this pattern "
                          f"({existing_clauses} clauses vs {new_clauses}) — skipping insert.")
                    return
                else:
                    # New rule is broader → it supersedes the old one
                    print(f"[PlaybookEditorAgent] New rule '{new_rule['id']}' is broader than '{r.get('id')}' "
                          f"({new_clauses} clauses vs {existing_clauses}) — replacing.")
                    rule_to_replace = r
                    break

        if rule_to_replace is not None:
            routing_rules.remove(rule_to_replace)

        if len(routing_rules) >= self.MAX_ROUTING_RULES:
            self._compact_rules(playbooks)

        # Insert at the TOP so patch rules override older / more general logic
        routing_rules.insert(0, new_rule)

        playbooks["learned_rule_audit"].append({
            "rule_id":      new_rule["id"],
            "event_id":     event_id,
            "trigger":      trigger,
            "appended_at":  new_rule["created_at"],
            "replaced":     rule_to_replace.get("id") if rule_to_replace else None
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
    def _derive_signature(telemetry: dict, threat_class: str = "unknown") -> str:
        """
        Produces a behaviour signature used as the memory key for allowlist/quarantine.

        threat_class is included to prevent ping-pong: without it, genuinely distinct
        traffic (e.g. a benign FIN teardown vs a Fuzzer attack) collapses to the same
        key (tcp_-_FIN), causing allowlist/quarantine to flip-flop on the same signature.
        Including threat_class means tcp_-_FIN_Normal and tcp_-_FIN_Fuzzers are tracked
        independently, so a patch rule for one never contaminates the other.
        """
        proto   = telemetry.get("proto",   "unknown")
        service = telemetry.get("service", "unknown")
        state   = telemetry.get("state",   "unknown")
        return f"{proto}_{service}_{state}_{threat_class}"

    # ------------------------------------------------------------------
    # Rule health I/O
    # ------------------------------------------------------------------

    def _load_rule_health(self) -> dict:
        if not os.path.exists(self.rule_health_file):
            return {}
        with open(self.rule_health_file, "r") as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}

    def _save_rule_health(self):
        with open(self.rule_health_file, "w") as f:
            json.dump(self.rule_health, f, indent=4)

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