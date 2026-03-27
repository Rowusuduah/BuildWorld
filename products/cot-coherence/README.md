# cot-coherence

**The first pip-installable tool that verifies whether your LLM's chain-of-thought reasoning is internally coherent.**

[![PyPI version](https://badge.fury.io/py/cot-coherence.svg)](https://badge.fury.io/py/cot-coherence)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## The Problem

LLMs routinely produce reasoning that *looks* correct but is incoherent:

- Steps that don't follow from each other (step gaps)
- Contradictions between intermediate steps
- Conclusions not entailed by the reasoning
- Unexplained logical leaps
- Certainty claimed that the steps don't support

This is especially dangerous when LLM reasoning is used as a CI/CD gate: automated code review, classification, safety checks. A reasoning chain that *appears* sound but is logically broken will produce confident wrong answers — and your gate won't catch it.

**Pydantic catches structural violations. DeepEval catches output quality. cot-coherence catches reasoning chain violations.**

*Biblical Pattern: Proverbs 14:12 — "There is a way that appears to be right, but in the end it leads to death."*

---

## Install

```bash
pip install cot-coherence
```

For YAML test suite support:
```bash
pip install "cot-coherence[yaml]"
```

---

## Quick Start

```python
from cot_coherence import check

report = check(
    steps=[
        "The experiment shows X correlates with Y.",
        "Correlation means causation.",   # ← logical leap
        "Therefore, increasing X will cause Y to increase.",
    ],
    conclusion="We should increase X to improve Y.",
)

print(report.status)          # INCOHERENT or DEGRADED
print(report.coherence_score) # e.g. 0.42
print(report.violations)      # [CoherenceViolation(REASONING_LEAP, severity=0.85, ...)]
```

Set `ANTHROPIC_API_KEY` in your environment. cot-coherence uses **Claude Haiku** by default — fast, cheap, designed for binary verdicts.

---

## Five Coherence Dimensions

| Dimension | What it checks |
|-----------|---------------|
| `step_continuity` | Does each step follow from the previous one? |
| `conclusion_grounding` | Does the conclusion follow from the last step(s)? |
| `internal_consistency` | Are all steps internally consistent with each other? |
| `reasoning_completeness` | Are critical intermediate steps missing? |
| `confidence_calibration` | Is claimed certainty warranted by what the steps establish? |

**Coherence score:** 1.00 = perfectly coherent, 0.00 = completely incoherent.
**Status:** `COHERENT` (≥0.80) · `DEGRADED` (0.55–0.79) · `INCOHERENT` (<0.55)

---

## Violation Types

| Type | Description |
|------|-------------|
| `STEP_GAP` | Step N doesn't follow from Step N-1 |
| `CONTRADICTION` | Two steps assert contradictory things |
| `UNSUPPORTED_CONCLUSION` | Conclusion not entailed by the reasoning |
| `REASONING_LEAP` | Critical intermediate step is missing |
| `OVERCONFIDENCE` | Certainty claimed not warranted by steps |
| `CIRCULAR` | Step merely restates a prior step |
| `SCOPE_SHIFT` | Reasoning suddenly shifts domain without justification |

---

## API

### `check()` — Simple function

```python
from cot_coherence import check

report = check(
    steps=["Step 1.", "Step 2.", "Step 3."],
    conclusion="Therefore X.",
    model="claude-haiku-4-5-20251001",  # default
    save=True,   # persist to SQLite trace log
)

print(report.status)             # CoherenceStatus.COHERENT
print(report.coherence_score)    # 0.87
print(report.incoherence_score)  # 0.13
print(report.to_markdown())      # Full markdown report
print(report.to_dict())          # JSON-serializable dict
```

**String input is auto-parsed:**

```python
report = check(
    steps="1. Observe X.\n2. Infer Y from X.\n3. Conclude Z from Y.",
    conclusion="Z is true.",
)
```

### `CoherenceChecker` — Reusable instance

```python
from cot_coherence import CoherenceChecker

checker = CoherenceChecker(model="claude-haiku-4-5-20251001")

# Single check
report = checker.check(steps=[...], conclusion="...")

# Batch check
results = checker.batch_check([
    {"steps": [...], "conclusion": "..."},
    {"steps": [...], "conclusion": "..."},
])
```

### `@coherence_check` — Decorator

```python
from cot_coherence import coherence_check, CoherenceError

@coherence_check(threshold=0.55, raise_on_fail=True)
def classify_with_reasoning(text: str) -> dict:
    # Call your LLM here...
    return {
        "steps": ["Step 1: ...", "Step 2: ...", "Step 3: ..."],
        "conclusion": "The text is positive.",
    }

try:
    result = classify_with_reasoning("some input")
    report = result["_coherence_report"]  # attached automatically
    print(f"Coherence: {report.coherence_score:.2f}")
except CoherenceError as e:
    print(f"Incoherent reasoning: {e}")
    print(e.report.violations)
```

---

## CI Gate — GitHub Actions

```yaml
- name: Check CoT coherence
  run: |
    cot-coherence check \
      --steps "Step 1: ..." "Step 2: ..." "Step 3: ..." \
      --conclusion "Therefore X." \
      --threshold 0.45
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

Exits `1` if `incoherence_score > threshold`. Exits `0` if coherent.

---

## YAML Test Suites

Define expected coherence for known reasoning chains:

```yaml
suite: "My CoT coherence suite"
threshold: 0.55
cases:
  - id: "valid_syllogism"
    steps:
      - "All men are mortal."
      - "Socrates is a man."
      - "Therefore, Socrates is mortal."
    conclusion: "Socrates is mortal."
    expect: "COHERENT"

  - id: "correlation_causation_leap"
    steps:
      - "X correlates with Y."
      - "Therefore X causes Y."
    conclusion: "Increasing X will increase Y."
    expect: "INCOHERENT"
```

```bash
cot-coherence suite my-suite.yaml
```

---

## CLI Reference

```bash
# Check coherence (separate steps)
cot-coherence check \
  --steps "Step 1." "Step 2." "Step 3." \
  --conclusion "Therefore X." \
  --format text|json|markdown \
  --threshold 0.45 \
  --no-save

# Check coherence (raw text, auto-parsed)
cot-coherence check \
  --raw-steps "1. First.\n2. Second.\n3. Third." \
  --conclusion "Therefore X."

# Run a YAML test suite
cot-coherence suite my-suite.yaml --format text|json

# Show recent history
cot-coherence history --n 20
```

---

## SQLite Trace Log

Every check is persisted to `.cot-coherence.db` (configurable via `COT_COHERENCE_DB` env var):

```python
from cot_coherence import load_recent_reports

rows = load_recent_reports(n=10)
for r in rows:
    print(r["timestamp"], r["status"], r["coherence_score"])
```

---

## Configuration

| Env var | Default | Description |
|---------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Required for LLM judge |
| `COT_COHERENCE_MODEL` | `claude-haiku-4-5-20251001` | Model for coherence judge |
| `COT_COHERENCE_DB` | `.cot-coherence.db` | SQLite trace log path |

---

## Why not DeepEval / ROSCOE?

| Tool | What it does | What it misses |
|------|-------------|----------------|
| **DeepEval** | Output quality (50+ metrics) | Step-chain coherence verification |
| **ROSCOE** (arXiv) | Research CoT metrics | Not pip-installable, no CI gate |
| **Pydantic** | Structural output validation | Semantic reasoning quality |
| **cot-coherence** | Reasoning chain coherence, 5 dimensions, CI gate | — |

---

## Open Core Pricing (Coming v0.3)

| Tier | Price | What's included |
|------|-------|----------------|
| **OSS** | Free forever | Full CLI + Python API + YAML suites |
| **Pro** | $29/month | Dashboard, team history, Slack alerts |
| **Enterprise** | $99/month | SSO, audit log, custom judge prompts, SLA |

---

## Built by BuildWorld

[BuildWorld](https://github.com/buildworld-ai) ships AI infrastructure tools.
Part of the developer tools suite: [model-parity](https://github.com/buildworld-ai/model-parity) · [spec-drift](https://github.com/buildworld-ai/spec-drift) · [drift-guard](https://github.com/buildworld-ai/drift-guard) · [llm-contract](https://github.com/buildworld-ai/llm-contract) · [prompt-lock](https://github.com/buildworld-ai/prompt-lock)

MIT License.
