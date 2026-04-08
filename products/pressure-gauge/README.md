# pressure-gauge

**Detect LLM behavioral drift caused by context window fill level.**

The first pip-installable **ContextPressureScore** for AI agents — measures "context anxiety": agents that rush to complete tasks, summarize prematurely, or change behavior as their context window fills.

[![PyPI version](https://img.shields.io/pypi/v/pressure-gauge.svg)](https://pypi.org/project/pressure-gauge/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/pressure-gauge/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

Long-running AI agents change behavior as their context window fills — a phenomenon called **"context anxiety"** in 2026 developer literature. Agents:

- Wrap up prematurely with "I'll summarize briefly..."
- Rush through steps they'd otherwise analyze carefully
- Falsely declare tasks complete when context is near capacity
- Produce lower-quality outputs at 90% fill vs. 10% fill

**This is invisible to standard observability tools.** No error is thrown. No timeout fires. The agent appears healthy in every dashboard. The only signal: behavioral drift between low-fill and high-fill outputs.

**Concrete evidence:**
- Research shows models exhibit "context anxiety" near window limits, prematurely summarizing or rushing completion (Redis Blog, 2026)
- "Context rot" degrades LLM output quality as input length grows — tested across 18 models, every model affected (Chroma Research, 2026)
- Nearly 65% of enterprise AI failures in 2025 attributed to context drift — not raw context exhaustion (Zylos Research, 2026)

## What existing tools miss

| Tool | ContextPressureScore | Drift Curve | Onset Token | CI Gate | Framework-Agnostic |
|------|:---:|:---:|:---:|:---:|:---:|
| **pressure-gauge** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Langfuse | ❌ | ❌ | ❌ | ❌ | ✅ |
| Arize Phoenix | ❌ | ❌ | ❌ | ❌ | ✅ |
| LangSmith | ❌ | ❌ | ❌ | ❌ | ❌ |
| Braintrust | ❌ | ❌ | ❌ | ❌ | ✅ |
| W&B Weave | ❌ | ❌ | ❌ | ❌ | ❌ |
| invariant-probe | ❌ | ❌ | ❌ | ✅ | ✅ |
| livelock-probe | ❌ | ❌ | ❌ | ✅ | ✅ |

*invariant-probe: environmental perturbation testing. livelock-probe: zero-progress detection. Different problems.*

---

## Installation

```bash
# Core (zero dependencies)
pip install pressure-gauge

# With CLI (click + rich)
pip install pressure-gauge[cli]

# With neural embeddings (sentence-transformers)
pip install pressure-gauge[neural]

# Everything
pip install pressure-gauge[full]
```

---

## Quick Start

```python
from pressure_gauge import PressureGauge, PressureConfig

# Configure
gauge = PressureGauge(PressureConfig(
    model_context_limit=8192,    # Your model's token limit
    criticality="HIGH",          # CRITICAL / HIGH / MEDIUM / LOW
))

# Your agent function: receives context string, returns response string
def my_agent(context: str) -> str:
    return llm.complete(context)

# Run the pressure sweep
report = gauge.sweep(
    agent_fn=my_agent,
    base_context="Analyze the document above and provide key insights.",
)

# Check results
print(report.summary())
# ContextPressureScore: 0.9234
# Verdict: STABLE
# Gate: PASSED
# Pressure onset: not detected
# Recommendation: Agent behavior is stable across context fill levels.

# CI gate
assert report.gate_passed, f"Context pressure: {report.recommendation}"
```

---

## How It Works

### Algorithm

1. **Fill levels**: Run the agent at 5 fill levels: 10%, 30%, 50%, 70%, 90% of context window
2. **Context padding**: At each level, pad the context to the target token count using lorem ipsum / fake conversation history
3. **Embedding**: Embed each output with TF-IDF cosine similarity (zero deps) or sentence-transformers (optional)
4. **ContextPressureScore**: Mean cosine similarity of non-baseline outputs vs. baseline (10% fill)
5. **Onset detection**: First fill level where similarity drops below `stability_threshold`
6. **CI gate**: `gate_passed = ContextPressureScore >= stability_threshold`

### Criticality thresholds

| Criticality | Min ContextPressureScore | Use when |
|-------------|--------------------------|----------|
| CRITICAL | 0.95 | Medical, legal, financial agents |
| HIGH | 0.85 | Production customer-facing agents |
| MEDIUM | 0.75 | Internal tooling, analysis agents |
| LOW | 0.65 | Exploratory, draft-quality agents |

---

## API

### PressureGauge

```python
from pressure_gauge import PressureGauge, PressureConfig, CriticalityLevel

# Full sweep (5 fill levels)
gauge = PressureGauge(PressureConfig(model_context_limit=8192))
report = gauge.sweep(agent_fn=my_agent, base_context="Solve the problem.")

# Quick sweep (3 levels: 10%, 50%, 90%) — faster CI check
report = gauge.quick(agent_fn=my_agent, base_context="task")

# Fine-grained onset estimation
onset_token = gauge.estimate_onset(agent_fn=my_agent, granularity=10)
```

### Decorator

```python
from pressure_gauge import pressure_probe

@pressure_probe(model_context_limit=8192, criticality="HIGH")
def my_agent(context: str) -> str:
    return llm.complete(context)

# Run normally
response = my_agent("my task")

# Run pressure sweep
report = my_agent.pressure_sweep(base_context="Analyze the document.")
assert report.gate_passed, report.recommendation
```

### pytest fixture

```python
def test_agent_stable_under_pressure(pressure_gauge_suite):
    pressure_gauge_suite.configure(
        model_context_limit=8192,
        fill_levels=[0.1, 0.5, 0.9],
        criticality="HIGH",
    )
    report = pressure_gauge_suite.sweep(my_agent, base_context="task")
    assert report.gate_passed, report.recommendation
```

### CLI

```bash
# Install with CLI extras
pip install pressure-gauge[cli]

# Quick demo
pgauge demo

# Run sweep
pgauge run --model-limit 8192 --criticality HIGH --context "Analyze the doc"

# Quick 3-level sweep
pgauge quick --model-limit 8192

# Gate check (for CI scripts)
pgauge gate --score 0.91 --criticality HIGH

# Estimate token counts
pgauge estimate --model-limit 128000 --criticality CRITICAL

# Fine-grained onset detection
pgauge onset --model-limit 8192 --granularity 20
```

---

## PressureReport

```python
@dataclass
class PressureReport:
    context_pressure_score: float    # 0.0–1.0 (higher = more stable)
    pressure_onset_token: Optional[int]   # Token count where drift starts
    verdict: DriftVerdict            # STABLE / MILD / MODERATE / SEVERE
    gate_passed: bool                # True = stable enough for criticality
    recommendation: str             # Human-readable guidance
    drift_curve: List[DriftPoint]   # Per-fill-level similarity data
    config: PressureConfig          # The config used for this run
```

---

## Known Limitations

- **KU-064: TF-IDF calibration.** Zero-dependency TF-IDF may not capture semantic drift in short outputs. Use `[neural]` extras with sentence-transformers for higher fidelity.
- **KU-065: Non-deterministic agents.** If the agent is stochastic, `runs_per_level > 1` averages outputs, but variance can confound drift detection. Use `runs_per_level=3` or more for stochastic agents.
- **KU-066: Padding realism.** Lorem ipsum padding doesn't match real conversation history. Use `padding_strategy="inject_history"` for more realistic simulated fill.
- **KU-067: Single-task bias.** ContextPressureScore measures stability on one `base_context`. Multi-task agents may need separate sweeps per task type.

---

## Connection to Other Tools

pressure-gauge fills a different gap than similar-sounding tools:

- **livelock-probe**: Is the agent making progress at all? (zero-progress detection)
- **invariant-probe**: Does the agent behave the same under environmental changes?
- **session-lens**: Does the agent remember prior conversation accurately?
- **pressure-gauge**: Does the agent behave the same regardless of how full its context is?

---

*Pattern source: PAT-078 — Daniel 5:5-6, 27 (The TEKEL Pressure Drift Pattern)*
*"You have been weighed on the scales and found wanting."*
