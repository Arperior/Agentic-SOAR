import os
import json
import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class TrustFactor:
    """A single contribution (positive or negative) to the trust score."""
    name: str
    delta: float          # negative = penalty, positive = bonus
    reason: str


@dataclass
class TrustEvaluation:
    """
    The structured output of PolicyAgent.evaluate().
    Passed downstream to ZeroTrustSOARAgent and persisted in memory.
    """
    trust_score: float                        # 0.0 (no trust) – 1.0 (full trust)
    behavior_signature: str                   # proto_service_state key
    threat_type: str
    factors: list = field(default_factory=list)   # list[TrustFactor]
    policy_violations: list = field(default_factory=list)  # list[str]
    is_acceptable: bool = False               # filled in by is_trust_acceptable()
    evaluated_at: str = ""

    def to_dict(self) -> dict:
        d = asdict(self)
        d["factors"] = [asdict(f) for f in self.factors]
        return d


# ---------------------------------------------------------------------------
# Default policy config (written to disk if policy_config.json is missing)
# ---------------------------------------------------------------------------

_DEFAULT_POLICY_CONFIG = {
    "_comment": "Edit this file to reflect your organisation's Zero Trust policy.",

    # --- Trust score decision threshold ---
    # This is the floor the ZeroTrust diamond tests against — completely
    # separate from the ML optimal_threshold used for the RCF gate.
    # The trust score starts at 1.0 and is reduced by additive penalties:
    #   ml_risk (up to 0.30) + threat_class (0.15–0.45) + proto_combo (0.10)
    #   + after_hours (0.10) + volume (0.10) + categorical_risk (0.10)
    # A clean low-risk Normal event with one minor violation accumulates
    # roughly 0.26+0.00+0.10+0.10 = 0.46 in penalties → score 0.54.
    # Setting the floor at 0.40 means: any event that clears the ML check
    # AND has a Normal threat class AND has ≤2 minor policy violations will
    # ALLOW without reaching the LLM.  Events with quarantine history,
    # high-severity threat classes, or high ML risk will still fail.
    # Raise toward 0.60 for a stricter posture; lower toward 0.25 for more
    # LLM bypass.  Do NOT copy the ML optimal_threshold here — that value
    # lives on a completely different 0-1 probability scale.
    "trust_score_threshold": 0.40,

    # --- ML risk weighting ---
    "ml_risk_weight": 0.30,          # how much fused_risk penalizes trust (0-1)
                                     # At 0.50 a fused_risk of 0.98 applies a 0.49
                                     # penalty alone, leaving almost no headroom for
                                     # threat-class, memory, time, and volume factors.
                                     # 0.30 caps the ML penalty at 0.30 and lets the
                                     # other deterministic signals contribute meaningfully.

    # --- Threat classification penalties ---
    # Keys must match attack_cat labels from your dataset.
    "threat_class_penalties": {
        "Normal":          0.00,
        "Generic":         0.20,
        "Exploits":        0.30,
        "Fuzzers":         0.20,
        "DoS":             0.15,      # handled by RATE_LIMIT_DOS, lighter penalty here
        "Reconnaissance":  0.25,
        "Backdoor":        0.40,
        "Analysis":        0.15,
        "Shellcode":       0.45,
        "Worms":           0.40,
        "UNKNOWN":         0.30       # fallback for unseen classes
    },

    # --- Memory-based penalties ---
    "quarantine_penalty":  0.40,      # signature is in active_quarantines
    "allowlist_bonus":     0.15,      # signature is in dynamic_allowlist (FP-verified)

    # --- Protocol / service whitelist ---
    # Traffic whose proto+service combo is NOT in this list gets a small penalty.
    # Empty list = whitelist disabled.
    "approved_proto_service_combos": [
        "tcp_http", "tcp_https", "tcp_ftp", "udp_dns",
        "tcp_smtp", "tcp_ssh", "tcp_rdp", "udp_ntp",
        "tcp_-", "udp_-", "icmp_-"   # FIX: Allow raw transport/network protocols to prevent instant 0.0 trust
    ],
    "unapproved_combo_penalty": 0.10,

    # --- Asset sensitivity tiers ---
    # Map dst_ip prefixes to a sensitivity multiplier applied ONLY to the
    # ML risk penalty — not to the full accumulated deficit.
    # Higher multiplier = harder penalisation of the raw ML risk signal.
    "asset_sensitivity_tiers": {
        "10.0.0.":   1.5,    # critical infrastructure
        "10.1.":     1.2,    # internal servers
        "192.168.":  1.0,    # standard workstations (baseline)
        "172.16.":   1.1,    # DMZ
        "DEFAULT":   1.0
    },

    # --- Time-of-day policy ---
    # Traffic outside business hours receives an additional penalty.
    # FIX: Expanded to 24 hours to accommodate global UNSW-NB15 dataset timezones
    "business_hours": {"start": 0, "end": 24},  
    "after_hours_penalty": 0.10,

    # --- Rate / volume anomaly ---
    # If spkts (source packets) exceeds this in a single event, add penalty.
    "high_volume_threshold": 5000,
    "high_volume_penalty": 0.10,

    # --- Anomaly score threshold for categorical risk ---
    # If the CatBoost categorical risk alone exceeds this, add a separate penalty.
    "high_categorical_risk_threshold": 0.70,
    "high_categorical_risk_penalty": 0.10,

    # --- Trust score floor / ceiling ---
    "score_floor": 0.0,
    "score_ceiling": 1.0
}


# ---------------------------------------------------------------------------
# PolicyAgent
# ---------------------------------------------------------------------------

class PolicyAgent:
    """
    Deterministic Zero Trust Policy Agent.

    Usage
    -----
    agent = PolicyAgent()                            # loads/creates policy_config.json
    trust_eval = agent.evaluate(context, memory)     # returns TrustEvaluation
    if agent.is_trust_acceptable(trust_eval, threshold=0.5):
        # fast-path ALLOW
    else:
        # route to LLM / SOAR response
    """

    def __init__(self, policy_config_file: str = "policy_config.json"):
        self.policy_config_file = policy_config_file
        self.config = self._load_config()

    # ------------------------------------------------------------------
    # Config I/O
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        if not os.path.exists(self.policy_config_file):
            print(f"[PolicyAgent] Creating default policy config: {self.policy_config_file}")
            with open(self.policy_config_file, "w") as f:
                json.dump(_DEFAULT_POLICY_CONFIG, f, indent=4)
            return dict(_DEFAULT_POLICY_CONFIG)

        with open(self.policy_config_file, "r") as f:
            cfg = json.load(f)

        # Backfill any missing keys from the default so old configs stay compatible
        changed = False
        for key, val in _DEFAULT_POLICY_CONFIG.items():
            if key not in cfg:
                cfg[key] = val
                changed = True
        if changed:
            with open(self.policy_config_file, "w") as f:
                json.dump(cfg, f, indent=4)

        return cfg

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, context: dict, memory: dict) -> TrustEvaluation:
        """
        Core evaluation method.

        Parameters
        ----------
        context : dict
            Output of ZeroTrustSOARAgent.construct_context().
            Expected keys: ml_risk_profile, predicted_threat_classification,
                           network_telemetry.
        memory : dict
            Current agent memory snapshot.
            Expected keys: active_quarantines (set/list), dynamic_allowlist (set/list).

        Returns
        -------
        TrustEvaluation
        """
        telemetry = context.get("network_telemetry", {})
        ml_profile = context.get("ml_risk_profile", {})
        threat_type = context.get("predicted_threat_classification", "UNKNOWN")
        behavior_signature = self._get_signature(telemetry, threat_type)
        event_timestamp = context.get("timestamp", datetime.datetime.now().isoformat())
        score = 1.0   # start at full trust; factors penalize downward
        factors: list[TrustFactor] = []
        violations: list[str] = []

        # 1. ML fused risk — primary signal, with asset sensitivity multiplier
        #    applied HERE so it amplifies only the ML risk penalty, not the
        #    full accumulated deficit (FIX: previously amplified everything).
        score, factors = self._apply_ml_risk(score, factors, ml_profile, telemetry)

        # 2. Threat classification
        score, factors, violations = self._apply_threat_class(
            score, factors, violations, threat_type
        )

        # 3. Memory: quarantine / allowlist
        score, factors = self._apply_memory_signals(
            score, factors, behavior_signature, memory
        )

        # 4. Protocol / service policy
        score, factors, violations = self._apply_proto_service_policy(
            score, factors, violations, telemetry
        )

        # 5. Time-of-day policy
        score, factors, violations = self._apply_time_policy(
            score, factors, violations, event_timestamp
        )

        # 6. Volume / rate anomaly
        score, factors, violations = self._apply_volume_policy(
            score, factors, violations, telemetry
        )

        # 7. Categorical risk spike
        score, factors = self._apply_categorical_risk(score, factors, ml_profile)

        # Clamp to [floor, ceiling]
        floor = self.config["score_floor"]
        ceiling = self.config["score_ceiling"]
        score = max(floor, min(ceiling, round(score, 6)))

        return TrustEvaluation(
            trust_score=score,
            behavior_signature=behavior_signature,
            threat_type=threat_type,
            factors=factors,
            policy_violations=violations,
            is_acceptable=False,        # set by is_trust_acceptable()
            evaluated_at=datetime.datetime.now().isoformat()
        )

    def is_trust_acceptable(
        self,
        trust_eval: TrustEvaluation,
        threshold: float = None
    ) -> bool:
        """
        The Zero Trust Evaluation 
        If threshold is omitted, resolves securely from policy_config.
        """
        # FIX: Resolve securely instead of hardcoding 0.5 in the signature
        if threshold is None:
            threshold = self.config.get("trust_score_threshold", 0.40)
            
        result = trust_eval.trust_score >= threshold
        trust_eval.is_acceptable = result
        return result

    # ------------------------------------------------------------------
    # Factor methods — each returns updated (score, factors[, violations])
    # ------------------------------------------------------------------

    def _apply_ml_risk(
        self, score: float, factors: list, ml_profile: dict, telemetry: dict
    ) -> tuple:
        """
        Applies the fused ML risk penalty, scaled by the asset sensitivity
        multiplier for the destination IP.

        FIX: The multiplier now applies ONLY to the ML risk penalty, not to
        the full accumulated trust deficit. This prevents non-linear collapse
        of the trust score on critical-subnet events that already have other
        penalties applied. The separate _apply_asset_sensitivity step has been
        removed and merged here so the scope of amplification is explicit.
        """
        fused = ml_profile.get("fused_risk", 0.5)
        weight = self.config["ml_risk_weight"]
        base_penalty = fused * weight

        # Resolve asset sensitivity multiplier
        tiers = self.config.get("asset_sensitivity_tiers", {})
        dst_ip = str(telemetry.get("dstip", ""))
        multiplier = tiers.get("DEFAULT", 1.0)
        matched_tier = "DEFAULT"
        for prefix, m in tiers.items():
            if prefix == "DEFAULT":
                continue
            if dst_ip.startswith(prefix):
                multiplier = m
                matched_tier = prefix
                break

        penalty = base_penalty * multiplier

        reason = f"fused_risk={fused:.4f} × weight={weight:.2f}"
        if multiplier != 1.0:
            reason += (
                f" × asset_multiplier={multiplier} (tier '{matched_tier}')"
                f" → base_penalty={base_penalty:.4f}, amplified={penalty:.4f}"
            )

        factors.append(TrustFactor(
            name="ml_fused_risk",
            delta=-penalty,
            reason=reason
        ))
        return score - penalty, factors

    def _apply_threat_class(
        self, score: float, factors: list, violations: list, threat_type: str
    ) -> tuple:
        penalties = self.config["threat_class_penalties"]
        penalty = penalties.get(threat_type, penalties.get("UNKNOWN", 0.30))

        if penalty > 0:
            factors.append(TrustFactor(
                name="threat_classification",
                delta=-penalty,
                reason=f"predicted_threat='{threat_type}' → penalty={penalty:.2f}"
            ))
            if penalty >= 0.35:
                violations.append(
                    f"HIGH_SEVERITY_THREAT: {threat_type} exceeds critical threshold"
                )

        return score - penalty, factors, violations

    def _apply_memory_signals(
        self, score: float, factors: list, sig: str, memory: dict
    ) -> tuple:
        quarantines = set(memory.get("active_quarantines", []))
        allowlist = set(memory.get("dynamic_allowlist", []))

        if sig in quarantines:
            penalty = self.config["quarantine_penalty"]
            factors.append(TrustFactor(
                name="quarantine_history",
                delta=-penalty,
                reason=f"signature '{sig}' is in active_quarantines"
            ))
            score -= penalty

        elif sig in allowlist:
            bonus = self.config["allowlist_bonus"]
            factors.append(TrustFactor(
                name="allowlist_bonus",
                delta=+bonus,
                reason=f"signature '{sig}' was previously verified as benign (FP)"
            ))
            score += bonus

        return score, factors

    def _apply_proto_service_policy(
        self, score: float, factors: list, violations: list, telemetry: dict
    ) -> tuple:
        approved = self.config.get("approved_proto_service_combos", [])
        if not approved:
            return score, factors, violations   # whitelist disabled

        proto = str(telemetry.get("proto", "")).lower()
        service = str(telemetry.get("service", "")).lower()
        combo = f"{proto}_{service}"

        if combo not in approved:
            penalty = self.config["unapproved_combo_penalty"]
            factors.append(TrustFactor(
                name="unapproved_proto_service",
                delta=-penalty,
                reason=f"combo '{combo}' not in approved_proto_service_combos"
            ))
            violations.append(f"UNAPPROVED_COMBO: {combo}")
            score -= penalty

        return score, factors, violations

    def _apply_time_policy(
        self, score: float, factors: list, violations: list, event_timestamp: str) -> tuple:
        hours = self.config.get("business_hours", {"start": 8, "end": 20})
        
        # FIX: Parse the hour directly from the event's ISO timestamp
        try:
            event_time = datetime.datetime.fromisoformat(event_timestamp)
            current_hour = event_time.hour
        except ValueError:
            current_hour = datetime.datetime.now().hour # Fallback if parsing fails

        if not (hours["start"] <= current_hour < hours["end"]):
            penalty = self.config["after_hours_penalty"]
            factors.append(TrustFactor(
                name="after_hours_access",
                delta=-penalty,
                reason=f"event at hour={current_hour}, outside business hours "
                       f"[{hours['start']}–{hours['end']})"
            ))
            violations.append(
                f"AFTER_HOURS_ACCESS: event at {current_hour:02d}:xx local time"
            )
            score -= penalty

        return score, factors, violations

    def _apply_volume_policy(
        self, score: float, factors: list, violations: list, telemetry: dict
    ) -> tuple:
        threshold = self.config.get("high_volume_threshold", 5000)
        penalty = self.config.get("high_volume_penalty", 0.10)

        spkts = int(telemetry.get("spkts", 0))
        if spkts > threshold:
            factors.append(TrustFactor(
                name="high_volume_anomaly",
                delta=-penalty,
                reason=f"spkts={spkts} > threshold={threshold}"
            ))
            violations.append(f"HIGH_VOLUME: spkts={spkts}")
            score -= penalty

        return score, factors, violations

    def _apply_categorical_risk(
        self, score: float, factors: list, ml_profile: dict
    ) -> tuple:
        cat_threshold = self.config.get("high_categorical_risk_threshold", 0.70)
        penalty = self.config.get("high_categorical_risk_penalty", 0.10)

        cat_risk = ml_profile.get("categorical_risk", 0.0)
        if cat_risk > cat_threshold:
            factors.append(TrustFactor(
                name="high_categorical_risk",
                delta=-penalty,
                reason=f"categorical_risk={cat_risk:.4f} > threshold={cat_threshold}"
            ))
            score -= penalty

        return score, factors

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_signature(telemetry: dict, threat_class: str = "unknown") -> str:
        proto = telemetry.get("proto", "unknown")
        service = telemetry.get("service", "unknown")
        state = telemetry.get("state", "unknown")
        return f"{proto}_{service}_{state}_{threat_class}"