"""
vector_memory.py
────────────────
Hybrid Vector Memory Layer for the AI-SOAR/Zero Trust system.

Architecture
────────────
┌─────────────────────────────────────────────────────────────────┐
│  JSON (keep as-is)          Vector Memory (new)                 │
│  ─────────────────          ────────────────────────────────    │
│  playbooks.json        ←──  synthesized_rules  collection       │
│  correction_log.json   ←──  correction_analyses collection      │
│  agent_memory.json          incident_summaries  collection      │
│                             synthesized_rules   collection      │
│                                                                  │
│  PolicyAgent (deterministic, unchanged)                          │
│  ZeroTrustSOARAgent  ──calls──►  VectorMemory.get_context()      │
│  PlaybookEditorAgent ──calls──►  VectorMemory.store_*()          │
└─────────────────────────────────────────────────────────────────┘

Collections
───────────
  correction_analyses  – root-cause analyses from every FP/FN correction.
                         Enables "have we seen this mistake before?" retrieval.
  incident_summaries   – per-event LLM reasoning + context digest.
                         Enables similar-incident lookup before a new LLM call.
  synthesized_rules    – every rule written by PlaybookEditorAgent, as natural
                         language.  Enables semantic rule deduplication and
                         context injection into the LLM prompt.

Usage
─────
  from vector_memory import VectorMemory

  vm = VectorMemory()                   # initialises ChromaDB + encoder once

  # Store after a correction
  vm.store_correction_analysis(record)  # CorrectionRecord or its dict form

  # Store after every LLM evaluate_incident call
  vm.store_incident_summary(event_id, context, decision)

  # Store when PlaybookEditorAgent writes a new rule
  vm.store_synthesized_rule(new_rule, event_id, correction_type)

  # Retrieve before an LLM call
  similar = vm.get_similar_incidents(context, n_results=3)
  prior   = vm.get_similar_corrections(context, n_results=3)
  rules   = vm.get_relevant_rules(context, n_results=5)

  # Build an enriched LLM context block (one-liner)
  extra_ctx = vm.build_llm_context_block(context)
"""

from __future__ import annotations

import json
import datetime
import hashlib
from typing import Optional, Union

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

_MODEL_NAME    = "all-MiniLM-L6-v2"
_CHROMA_DIR    = "chroma_db"                 # persisted on disk next to your saves
_MAX_RESULTS   = 5                           # default retrieval limit per query


# ─────────────────────────────────────────────────────────────────────────────
# Text serialisers  (context → query string)
# ─────────────────────────────────────────────────────────────────────────────

def _context_to_text(context: dict) -> str:
    """
    Convert a ZeroTrustSOARAgent context dict to a short natural-language
    description used as the embedding query / document text.

    Focuses on the fields the LLM reasoning actually references so that
    semantic similarity aligns with decision similarity.
    """
    ml   = context.get("ml_risk_profile", {})
    tel  = context.get("network_telemetry", {})
    tc   = context.get("predicted_threat_classification", "unknown")

    return (
        f"threat={tc} "
        f"fused_risk={ml.get('fused_risk', '?')} "
        f"cat_risk={ml.get('categorical_risk', '?')} "
        f"anomaly_risk={ml.get('anomaly_risk', '?')} "
        f"proto={tel.get('proto', '?')} "
        f"service={tel.get('service', '?')} "
        f"state={tel.get('state', '?')} "
        f"spkts={tel.get('spkts', '?')} "
        f"dpkts={tel.get('dpkts', '?')}"
    )


def _rule_to_text(rule: Union[dict, str]) -> str:
    """Convert a routing rule (dict or string) to embeddable text."""
    if isinstance(rule, str):
        return rule
    return (
        f"Rule {rule.get('id', '?')}: "
        f"IF {rule.get('condition', '?')} "
        f"THEN {rule.get('playbook', '?')}. "
        f"Rationale: {rule.get('rationale', 'n/a')}"
    )


def _stable_id(prefix: str, *parts) -> str:
    """
    Deterministic document ID so re-storing the same event doesn't create
    duplicate embeddings.  Hashes the concatenated string parts.
    """
    raw = prefix + "|" + "|".join(str(p) for p in parts)
    return prefix + "_" + hashlib.md5(raw.encode()).hexdigest()[:12]


# ─────────────────────────────────────────────────────────────────────────────
# VectorMemory
# ─────────────────────────────────────────────────────────────────────────────

class VectorMemory:
    """
    Thin wrapper around ChromaDB + sentence-transformers that provides
    domain-specific store / retrieve methods for the SOAR pipeline.

    Thread safety: ChromaDB's PersistentClient is not thread-safe for
    concurrent writes.  For a single-process evaluation loop this is fine.
    Add a threading.Lock if you parallelise event processing.
    """

    def __init__(
        self,
        chroma_dir:  str = _CHROMA_DIR,
        model_name:  str = _MODEL_NAME,
    ):
        print(f"[VectorMemory] Initialising ChromaDB at '{chroma_dir}' …")
        self._client = chromadb.PersistentClient(
            path=chroma_dir,
            settings=Settings(anonymized_telemetry=False)
        )

        print(f"[VectorMemory] Loading encoder '{model_name}' …")
        self._encoder = SentenceTransformer(model_name)

        # Three domain collections — created if they don't exist yet
        self._col_corrections = self._client.get_or_create_collection(
            name="correction_analyses",
            metadata={"hnsw:space": "cosine"},
        )
        self._col_incidents = self._client.get_or_create_collection(
            name="incident_summaries",
            metadata={"hnsw:space": "cosine"},
        )
        self._col_rules = self._client.get_or_create_collection(
            name="synthesized_rules",
            metadata={"hnsw:space": "cosine"},
        )

        print(
            f"[VectorMemory] Ready.  "
            f"corrections={self._col_corrections.count()}  "
            f"incidents={self._col_incidents.count()}  "
            f"rules={self._col_rules.count()}"
        )

    # ──────────────────────────────────────────────────────────────────────
    # Encode helper
    # ──────────────────────────────────────────────────────────────────────

    def _embed(self, text: str) -> list[float]:
        """Return a single embedding vector for the given text."""
        return self._encoder.encode(text, normalize_embeddings=True).tolist()

    # ──────────────────────────────────────────────────────────────────────
    # STORE  —  called by PlaybookEditorAgent / ZeroTrustSOARAgent
    # ──────────────────────────────────────────────────────────────────────

    def store_correction_analysis(
        self,
        record,                    # CorrectionRecord instance OR its .to_dict()
        correction_type: str = "unknown",   # "FP" or "FN"
    ) -> None:
        """
        Embeds and stores a CorrectionRecord's analysis text so future events
        can retrieve "have we diagnosed this type of mistake before?".

        Accepts either the CorrectionRecord object or a plain dict so it can
        be called from both the live agent and a post-hoc batch loader.
        """
        d = record.to_dict() if hasattr(record, "to_dict") else record

        event_id = str(d.get("event_id", "?"))
        analysis = d.get("analysis", "")
        context  = d.get("context",  {})

        if not analysis or "LLM unavailable" in analysis:
            return     # no useful signal to embed

        doc_id   = _stable_id("corr", event_id, analysis[:40])
        doc_text = (
            f"[{correction_type}] Event {event_id}: {analysis} | "
            f"context: {_context_to_text(context)}"
        )

        self._col_corrections.upsert(
            ids        = [doc_id],
            embeddings = [self._embed(doc_text)],
            documents  = [doc_text],
            metadatas  = [{
                "event_id":        event_id,
                "correction_type": correction_type,
                "corrected_at":    d.get("corrected_at", datetime.datetime.now().isoformat()),
                "new_rule_id":     (d.get("new_rule") or {}).get("id", "none"),
            }],
        )
        print(f"[VectorMemory] Stored correction analysis for event {event_id} → {doc_id}")

    def store_incident_summary(
        self,
        event_id,
        context: dict,
        decision: dict,
    ) -> None:
        """
        Embeds and stores a summary of the incident context + LLM decision.
        Retrieved before new LLM calls to inject "similar past decisions"
        as few-shot examples.
        """
        event_id = str(event_id)
        ctx_text = _context_to_text(context)
        playbook = decision.get("playbook", "?")
        reasoning = decision.get("reasoning", "")[:200]

        doc_id   = _stable_id("inc", event_id, ctx_text[:40])
        doc_text = (
            f"Event {event_id} | {ctx_text} | "
            f"Decision: {playbook} | Reasoning: {reasoning}"
        )

        self._col_incidents.upsert(
            ids        = [doc_id],
            embeddings = [self._embed(doc_text)],
            documents  = [doc_text],
            metadatas  = [{
                "event_id":  event_id,
                "playbook":  playbook,
                "timestamp": context.get("timestamp", datetime.datetime.now().isoformat()),
                "fused_risk": str(context.get("ml_risk_profile", {}).get("fused_risk", "?")),
                "threat":    context.get("predicted_threat_classification", "?"),
            }],
        )
        print(f"[VectorMemory] Stored incident summary for event {event_id} → {doc_id}")

    def store_synthesized_rule(
        self,
        rule: dict,
        event_id,
        correction_type: str = "unknown",
    ) -> None:
        """
        Embeds and stores a newly synthesised routing rule.
        Used for semantic deduplication and context injection into the LLM prompt.
        """
        rule_id  = rule.get("id", _stable_id("rule", str(event_id)))
        doc_text = _rule_to_text(rule)
        doc_id   = _stable_id("rule", rule_id)

        self._col_rules.upsert(
            ids        = [doc_id],
            embeddings = [self._embed(doc_text)],
            documents  = [doc_text],
            metadatas  = [{
                "rule_id":         rule_id,
                "event_id":        str(event_id),
                "correction_type": correction_type,
                "playbook":        rule.get("playbook", "?"),
                "created_at":      rule.get("created_at", datetime.datetime.now().isoformat()),
            }],
        )
        print(f"[VectorMemory] Stored synthesized rule '{rule_id}' → {doc_id}")

    # ──────────────────────────────────────────────────────────────────────
    # RETRIEVE  —  called by ZeroTrustSOARAgent / PlaybookEditorAgent
    # ──────────────────────────────────────────────────────────────────────

    def get_similar_incidents(
        self,
        context: dict,
        n_results: int = 3,
    ) -> list[dict]:
        """
        Returns the N most semantically similar past incidents to the current
        context.  Each result has keys: document, metadata, distance.
        """
        if self._col_incidents.count() == 0:
            return []

        query = _context_to_text(context)
        results = self._col_incidents.query(
            query_embeddings = [self._embed(query)],
            n_results        = min(n_results, self._col_incidents.count()),
            include          = ["documents", "metadatas", "distances"],
        )
        return _unpack_results(results)

    def get_similar_corrections(
        self,
        context: dict,
        n_results: int = 3,
    ) -> list[dict]:
        """
        Returns the N most semantically similar past correction analyses.
        Useful for PlaybookEditorAgent to check "did we already write a rule
        for this type of mistake?"
        """
        if self._col_corrections.count() == 0:
            return []

        query = _context_to_text(context)
        results = self._col_corrections.query(
            query_embeddings = [self._embed(query)],
            n_results        = min(n_results, self._col_corrections.count()),
            include          = ["documents", "metadatas", "distances"],
        )
        return _unpack_results(results)

    def get_relevant_rules(
        self,
        context: dict,
        n_results: int = 5,
    ) -> list[dict]:
        """
        Returns the N most semantically relevant synthesized rules for the
        given context.  Injected into the LLM prompt as learned routing guidance.
        """
        if self._col_rules.count() == 0:
            return []

        query = _context_to_text(context)
        results = self._col_rules.query(
            query_embeddings = [self._embed(query)],
            n_results        = min(n_results, self._col_rules.count()),
            include          = ["documents", "metadatas", "distances"],
        )
        return _unpack_results(results)

    # ──────────────────────────────────────────────────────────────────────
    # COMPOSITE  — one-liner LLM context enrichment
    # ──────────────────────────────────────────────────────────────────────

    def build_llm_context_block(
        self,
        context: dict,
        max_incidents:   int = 2,
        max_corrections: int = 2,
        max_rules:       int = 3,
        similarity_threshold: float = 0.75,   # cosine distance ceiling
    ) -> str:
        """
        Retrieves similar incidents, past corrections, and relevant learned
        rules, then formats them into a compact string block suitable for
        injection into the LLM system prompt.

        Only includes results whose cosine distance < similarity_threshold
        so low-quality matches don't pollute the context window.

        Returns empty string when the collections are empty (first run).
        """
        parts: list[str] = []

        # ── Similar incidents ──────────────────────────────────────────
        incidents = [
            r for r in self.get_similar_incidents(context, n_results=max_incidents)
            if r["distance"] < similarity_threshold
        ]
        if incidents:
            parts.append("SIMILAR PAST INCIDENTS (for reference, not binding rules):")
            for i, r in enumerate(incidents, 1):
                m = r["metadata"]
                parts.append(
                    f"  {i}. Event {m.get('event_id')} → {m.get('playbook')} "
                    f"(risk={m.get('fused_risk')}, threat={m.get('threat')}, "
                    f"similarity={1 - r['distance']:.2f}): {r['document'][:160]}"
                )

        # ── Past corrections ───────────────────────────────────────────
        corrections = [
            r for r in self.get_similar_corrections(context, n_results=max_corrections)
            if r["distance"] < similarity_threshold
        ]
        if corrections:
            parts.append("\nKNOWN MISTAKE PATTERNS (diagnoses of prior FP/FN errors):")
            for i, r in enumerate(corrections, 1):
                m = r["metadata"]
                parts.append(
                    f"  {i}. [{m.get('correction_type')}] {r['document'][:180]}"
                )

        # ── Relevant learned rules ─────────────────────────────────────
        rules = [
            r for r in self.get_relevant_rules(context, n_results=max_rules)
            if r["distance"] < similarity_threshold
        ]
        if rules:
            parts.append("\nRELEVANT LEARNED RULES (AI-synthesised, lower priority than native rules):")
            for i, r in enumerate(rules, 1):
                parts.append(f"  {i}. {r['document'][:200]}")

        return "\n".join(parts)

    def stats(self) -> dict:
        return {
            "correction_analyses": self._col_corrections.count(),
            "incident_summaries":  self._col_incidents.count(),
            "synthesized_rules":   self._col_rules.count(),
        }

    def delete_rule(self, rule_id: str) -> None:
        """
        Removes a synthesized rule embedding from ChromaDB by its rule_id
        (e.g. 'LEARNED_001', 'FN_PATCH_002').

        Called by PlaybookEditorAgent.consolidate_rules() after merging
        rules so stale embeddings never get injected into LLM prompts.
        Safe to call on a rule_id that doesn't exist — the except block
        silently ignores the miss rather than crashing consolidation.

        Parameters
        ----------
        rule_id : str
            The rule's id field as written by PlaybookEditorAgent,
            e.g. 'LEARNED_001'. Internally converted to the stable
            ChromaDB document ID using the same hash used at store time.
        """
        doc_id = _stable_id("rule", rule_id)
        try:
            self._col_rules.delete(ids=[doc_id])
            print(f"[VectorMemory] Deleted rule embedding '{rule_id}' → {doc_id}")
        except Exception:
            # Rule was never stored, already deleted, or ChromaDB miss —
            # all are safe to ignore during consolidation.
            pass


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _unpack_results(results: dict) -> list[dict]:
    """Flatten ChromaDB query results into a list of dicts."""
    out = []
    docs      = (results.get("documents")  or [[]])[0]
    metas     = (results.get("metadatas")  or [[]])[0]
    distances = (results.get("distances")  or [[]])[0]
    for doc, meta, dist in zip(docs, metas, distances):
        out.append({"document": doc, "metadata": meta, "distance": dist})
    return out