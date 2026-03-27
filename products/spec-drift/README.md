# spec-drift

**Your LLM outputs pass Pydantic. That's not enough.**

spec-drift detects when your LLM outputs drift semantically from their declared specification — even when structural validation passes. It monitors continuous semantic compliance in production, generates drift reports, and provides a CI gate for semantic regression testing.

---

## The Problem

Every team shipping LLM features uses Pydantic or JSON Schema to validate output structure. These tools are excellent — and completely blind to semantic drift.

Consider what happens in production LLM systems over time:

**Silent model updates:** Your LLM provider silently updates the underlying model. The API contract (field names, types, schema) doesn't change. But the distribution of values inside those fields shifts. Your "sentiment" field starts returning "ambivalent" where it previously returned "neutral." Your "risk_level" classification shifts its decision boundary. Pydantic sees nothing.

**Prompt erosion:** A prompt is modified through six iterations of "just a small tweak." Each tweak passes regression tests individually. But cumulatively, the semantic profile of outputs drifts. The "reasoning" field that used to average 120 words now averages 30. Your validation still passes.

**Input distribution shift:** A new user cohort or marketing campaign brings different input patterns. The same model, same prompt, same schema — but outputs drift from the spec because they were calibrated for a different input distribution.

spec-drift catches all of these. Pydantic catches none of them.

---

## The Biblical Foundation

*"Moses then said to Aaron, 'This is what the Lord spoke of when he said: Among those who approach me I will be proved holy.'"* — Leviticus 10:3

Nadab and Abihu offered "unauthorized fire" — structurally correct (fire), instrumentally correct (censers), personally authorized (priests). But the semantic specification was violated. Every structural check passed. The semantic compliance check failed.

spec-drift applies this principle to LLM outputs: structural validation is necessary but not sufficient. Semantic specification must be declared and continuously monitored.

*BibleWorld build — PAT-037, Pivot_Score 8.63*

---

## Installation

```bash
pip install spec-drift
```

**Requirements:** Python 3.10+, Pydantic v2

---

## Quick Start

### 1. Declare a semantic spec

```python
from pydantic import BaseModel
from spec_drift import spec, SemanticConstraint

@spec(
    category=SemanticConstraint.from_authorized_values(
        ["positive", "negative", "neutral"],
        tolerance=0.02,       # max 2% outputs outside authorized set
        alert_threshold=0.10  # alert if >10% observations violate
    ),
    reasoning=SemanticConstraint.from_length_bounds(
        min_words=30,
        max_words=300,
        alert_threshold=0.15
    ),
    score=SemanticConstraint.from_distribution(
        mean=6.5,
        std=2.0,
        drift_threshold=1.0,  # alert if mean shifts >1 sigma
        alert_threshold=0.20
    )
)
class SentimentAnalysis(BaseModel):
    category: str
    reasoning: str
    score: float
```

### 2. Wrap your LLM function

```python
from spec_drift import DriftMonitor
import anthropic

client = anthropic.Anthropic()

monitor = DriftMonitor(
    spec=SentimentAnalysis,
    db_path="./spec_drift.db",
    model_version="claude-3-5-haiku-20241022",
)

@monitor.watch
def analyze_sentiment(text: str) -> SentimentAnalysis:
    response = client.messages.create(
        model="claude-3-5-haiku-20241022",
        max_tokens=500,
        messages=[{"role": "user", "content": f"Analyze: {text}"}]
    )
    return SentimentAnalysis.model_validate_json(response.content[0].text)
```

### 3. Check drift

```bash
# Terminal drift report (last 7 days)
spec-drift check --spec my_module.SentimentAnalysis --since 7d

# CI gate: fail build if >20% semantic violations
spec-drift ci \
  --spec my_module.SentimentAnalysis \
  --test-batch data/ci_batch.jsonl \
  --threshold 0.20 \
  --exit-code
```

---

## API Reference

### `@spec(**constraints)`

Attaches semantic constraints to a Pydantic model class.

```python
@spec(field_name=SemanticConstraint.from_authorized_values([...]))
class MyModel(BaseModel):
    field_name: str
```

### `SemanticConstraint`

#### `SemanticConstraint.from_authorized_values(authorized, tolerance, alert_threshold)`
Field values must be drawn from the authorized list (within tolerance).
- `authorized`: list of permitted values
- `tolerance`: float, max fraction of outputs outside authorized set before constraint flags
- `alert_threshold`: float, fraction of rolling observations before alert fires

#### `SemanticConstraint.from_length_bounds(min_words, max_words, alert_threshold)`
String field word count must be within [min_words, max_words].

#### `SemanticConstraint.from_distribution(mean, std, drift_threshold, alert_threshold)`
Numeric field should follow a distribution near (mean, std). Alerts if observed mean shifts by more than drift_threshold standard deviations.

#### `SemanticConstraint.from_pattern(regex, min_match_rate, alert_threshold)`
String field should match the regex pattern at min_match_rate frequency.

### `DriftMonitor(spec, db_path, model_version, prompt_hash, alert_callback)`

Runtime monitor for semantic specification compliance.

#### `.watch` (decorator)
Wraps an LLM function to automatically observe its return value.

#### `.observe(output) -> output`
Manually observe a Pydantic model instance. Returns the output unchanged.

#### `.drift_report(since_hours) -> dict`
Generate a semantic drift report for the last N hours.

```python
{
    "spec": "SentimentAnalysis",
    "period_hours": 168.0,
    "observations": 4523,
    "violation_rate": 0.0312,
    "severity": "low",
    "field_violation_rates": {
        "category": 0.0089,
        "reasoning": 0.0221,
        "score": 0.0002
    }
}
```

#### `run_ci_gate(monitor, test_outputs, threshold) -> (passed, report)`
Run a CI gate on a batch of test outputs. Returns (passed, report).

---

## CLI Reference

```bash
# Initialize spec-drift in a project
spec-drift init

# Calibrate a baseline from golden data
spec-drift calibrate \
  --spec my_module.SentimentAnalysis \
  --input-file data/golden_set.jsonl \
  --output baseline.db

# Drift report (table format, last 7 days)
spec-drift check \
  --spec my_module.SentimentAnalysis \
  --since 7d \
  --format table

# CI gate
spec-drift ci \
  --spec my_module.SentimentAnalysis \
  --test-batch data/ci_batch.jsonl \
  --threshold 0.20 \
  --exit-code

# Compare two model versions
spec-drift compare \
  --spec my_module.SentimentAnalysis \
  --baseline-a baseline_gpt4o.db \
  --baseline-b baseline_claude_haiku.db

# HTML report
spec-drift report \
  --spec my_module.SentimentAnalysis \
  --since 30d \
  --output report.html
```

---

## GitHub Action

```yaml
# .github/workflows/llm-spec-check.yml
name: LLM Semantic Spec Check

on: [push, pull_request]

jobs:
  spec-drift:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: spec-drift/action@v1
        with:
          spec: my_module.SentimentAnalysis
          test-batch: data/ci_batch.jsonl
          threshold: '0.20'
```

---

## Severity Levels

| Violation Rate | Severity | Recommended Action |
|---|---|---|
| 0% | NONE | No action needed |
| < 5% | LOW | Monitor, no immediate action |
| 5-15% | MEDIUM | Investigate, create issue |
| 15-30% | HIGH | Rollback or prompt fix recommended |
| > 30% | CRITICAL | Immediate rollback |

---

## Storage

spec-drift uses SQLite by default — zero infrastructure required.

```python
# Local development (default)
monitor = DriftMonitor(spec=MyModel, db_path="./spec_drift.db")

# PostgreSQL for production (coming in v0.2)
monitor = DriftMonitor(
    spec=MyModel,
    db_url="postgresql://user:pass@host/db"
)

# In-memory for testing
monitor = DriftMonitor(spec=MyModel, db_path=":memory:")
```

---

## Roadmap

### v0.1 (this release)
- Core `@spec` decorator + `SemanticConstraint` DSL
- `DriftMonitor` with `.watch` and `.observe`
- SQLite observation store
- `run_ci_gate` function
- CLI: `check`, `ci`, `compare`

### v0.2
- PostgreSQL support
- Multi-field correlation monitoring
- Automatic model version detection (via LLM API response headers)
- Slack/PagerDuty alert integrations
- HTML drift reports

### v0.3
- LLM-judge semantic constraint evaluation (for complex, prose-level constraints)
- Baseline versioning with SemVer
- Team dashboard (hosted cloud option)
- Prometheus/Grafana metrics export

---

## Comparison

| Tool | Structural validation | Semantic spec monitoring | Production continuous | CI gate | Open source |
|------|---|---|---|---|---|
| Pydantic | YES | NO | NO | NO | YES |
| DeepEval | No (batch eval) | YES (point-in-time) | NO | YES | YES |
| Evidently | No (statistical drift) | NO | YES | NO | YES |
| Langfuse | NO | NO | YES (observability) | NO | YES |
| **spec-drift** | YES (via Pydantic) | **YES** | **YES** | **YES** | **YES** |

---

## Contributing

spec-drift is MIT licensed. Contributions welcome.

```bash
git clone https://github.com/bibleworld/spec-drift
cd spec-drift
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT — free to use, modify, and distribute.

---

*Built with BibleWorld — Pattern: Leviticus 10:1-3 (The Authorized Fire)*
*"Among those who approach me I will be proved holy." — Leviticus 10:3*
