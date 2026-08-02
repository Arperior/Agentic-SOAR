# Agentic SOAR

Agentic SOAR is a risk-adaptive Security Orchestration, Automation, and Response (SOAR) framework that combines machine learning, anomaly detection, and agentic reasoning to automate incident triage and enforce Zero Trust policies.

Unlike traditional SOAR systems that rely on static playbooks and binary allow/block decisions, Agentic SOAR continuously evaluates risk signals and orchestrates proportional responses through a multi-agent architecture.

---

## Motivation

Modern Security Operations Centers (SOCs) generate massive volumes of telemetry, making manual incident triage increasingly impractical. Existing SOAR platforms mitigate some of this burden but remain constrained by deterministic rules, resulting in:

- Alert fatigue
- Static policy enforcement
- Poor adaptation to novel attacks
- Inability to reason under uncertainty

Agentic SOAR addresses these limitations by combining probabilistic anomaly detection, supervised learning, and autonomous reasoning within a Zero Trust framework. :contentReference[oaicite:1]{index=1}

---

## Key Contributions

- Designed a multi-agent SOAR pipeline for adaptive incident response.
- Combined supervised threat classification and unsupervised anomaly detection using a cost-sensitive meta-learner.
- Implemented continuous Zero Trust policy enforcement using dynamic trust scores.
- Developed a self-healing feedback loop capable of synthesizing new remediation rules from false positives and false negatives.
- Evaluated the framework on the UNSW-NB15 intrusion detection benchmark.

---

## System Architecture

```text
                         Network Telemetry
                                 │
                                 ▼

                ┌──────────────────────────────┐
                │   Threat Detection Layer     │
                │                              │
                │ • Binary Classification      │
                │ • Incident Classification    │
                │ • Anomaly Detection          │
                └──────────────┬───────────────┘
                               │
                               ▼

                ┌──────────────────────────────┐
                │      Risk Fusion Layer       │
                │                              │
                │ • Threat Probability         │
                │ • Anomaly Score              │
                │ • Prediction Entropy         │
                └──────────────┬───────────────┘
                               │
                               ▼

                ┌──────────────────────────────┐
                │   Zero Trust Policy Agent    │
                │                              │
                │ • Trust Computation          │
                │ • Policy Enforcement         │
                └──────────────┬───────────────┘
                               │
                               ▼

                ┌──────────────────────────────┐
                │      Response Agent          │
                │                              │
                │ • Threat Reasoning           │
                │ • Action Selection           │
                └──────────────┬───────────────┘
                               │
                               ▼

                ┌──────────────────────────────┐
                │ Automated Response Actions   │
                │                              │
                │ • Step-up Authentication     │
                │ • Rate Limiting              │
                │ • Quarantine                 │
                │ • Network Isolation          │
                └──────────────┬───────────────┘
                               │
                               ▼

                      Self-Healing Feedback
```

---

## Pipeline Overview

### 1. Threat Detection

The first stage analyzes incoming telemetry using:

- Binary threat classification
- Multi-class incident classification
- Random Cut Forest anomaly detection

---

### 2. Risk Fusion

A cost-sensitive meta-learner combines:

- Threat probabilities
- Anomaly scores
- Classification uncertainty

to produce a unified risk score.

---

### 3. Zero Trust Enforcement

The policy layer continuously recomputes trust using:

- Historical behavior
- Access patterns
- Network telemetry
- Quarantine history

Instead of binary decisions, responses scale proportionally with risk.

---

### 4. Agentic Response

The response agent reasons over:

- Threat category
- Risk score
- Trust score
- Organizational constraints

and selects mitigation actions accordingly.

---

### 5. Self-Healing

The framework continuously audits operational errors and:

- analyzes false positives
- analyzes false negatives
- generates remediation rules
- consolidates redundant policies

---

## Technical Stack

| Category | Technologies |
|---|---|
| Machine Learning | CatBoost, Scikit-learn |
| Anomaly Detection | Random Cut Forest |
| Meta Learning | Logistic Regression |
| AI | LLM-based agents |
| Security | SOAR, Zero Trust |
| Dataset | UNSW-NB15 |

---

## Evaluation

The framework is evaluated on the UNSW-NB15 benchmark using:

### Detection Metrics

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

### Operational Metrics

- Alert reduction rate
- False-positive rate
- False-negative rate
- Mean time to detect (MTTD)
- Mean time to respond (MTTR)

### Agent Metrics

- Policy-routing accuracy
- Response latency
- Rule synthesis success rate
- Self-healing effectiveness

The evaluation compares standalone models against the complete agentic pipeline under distribution shifts and class imbalance scenarios. :contentReference[oaicite:2]{index=2}

---

## Results

| Component | Accuracy | F1 | ROC-AUC |
|---|---:|---:|---:|
| Binary Classifier | 93 | 0.93 | 0.9684 |
| MultiClass Classifier | 83 | 0.83 | 0.9684 |
| Random Cut Forest | 74 | 0.69 | 0.7471 |
| Meta-Learner | 81 | 0.8019 | - |
| Agentic SOAR Pipeline | 93 | 0.92 | - |


---


## Future Work

- Human-in-the-loop verification
- Cloud-native SIEM integration
- Dynamic recalibration for concept drift
- Distributed agent execution

---

## Authors

Arib Ansari

Ahal Khan

Department of Computer Engineering, Jamia Millia Islamia
