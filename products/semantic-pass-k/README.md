# semantic-pass-k

**Is your AI agent consistent? Run it k times. Find out.**

[![PyPI](https://img.shields.io/pypi/v/semantic-pass-k)](https://pypi.org/project/semantic-pass-k/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

AI agents are non-deterministic. The same task, run 5 times, gives 5 different outputs. Some are semantically equivalent. Some are not.

How do you know if your agent is *consistent enough* for the task at hand?

- A medical diagnosis agent must be near-perfectly consistent (>99% semantic similarity across runs).
- A customer-facing support agent should be highly consistent (>90%).
- An internal brainstorming tool can tolerate moderate variation (>60%).

No existing tool gives you a **ConsistencyScore** with **criticality-tier thresholds** and a **CI gate**. `semantic-pass-k` does.

---

## How It Works

1. Run your agent **k times** on the same prompt
2. Compute **pairwise cosine similarity** across all k outputs
3. `ConsistencyScore` = mean pairwise similarity
4. Compare against your **criticality tier** threshold
5. **CONSISTENT / BORDERLINE / INCONSISTENT** verdict

```
k=5 runs → 10 pairwise comparisons → ConsistencyScore → CI gate
```

### Criticality Tiers

| Tier | Threshold | Use Case |
|------|-----------|----------|
| CRITICAL | 0.99 | Medical, legal, financial |
| HIGH | 0.90 | Customer-facing, production agents |
| MEDIUM | 0.75 | Internal tools, best-effort |
| LOW | 0.60 | Exploration, brainstorming |

---

## Install

```bash
pip install semantic-pass-k          # zero-dependency core (TF-IDF similarity)
pip install "semantic-pass-k[neural]"  # sentence-transformers for better embeddings
pip install "semantic-pass-k[cli]"    # CLI tools (click + rich)
pip install "semantic-pass-k[full]"   # everything
```

---

## Quick Start

### Python API

```python
from semantic_pass_k import ConsistencyEngine

engine = ConsistencyEngine()

result = engine.evaluate(
    prompt="Summarise Ghana's Investment Policy 2024.",
    outputs=[
        "Ghana's Investment Policy prioritises local content...",
        "The policy focuses on local content requirements...",
        "Key emphasis is on local participation in investments...",
        "Ghana's 2024 policy centres on local content rules...",
        "The investment framework requires local content compliance...",
    ],
    criticality="HIGH",
)

print(result.verdict)           # CONSISTENT
print(result.consistency_score) # 0.94
print(result.threshold)         # 0.90
```

### ConsistencyRunner (automatic k runs)

```python
from semantic_pass_k import ConsistencyRunner
import anthropic

client = anthropic.Anthropic()

def my_agent(prompt: str) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text

runner = ConsistencyRunner(agent_fn=my_agent, k=5, criticality="HIGH")
result = runner.run("What is the capital of Ghana?")

print(result.verdict)   # CONSISTENT / BORDERLINE / INCONSISTENT
print(result.consistency_score)
```

### Batch Evaluation

```python
report = runner.run_batch(
    prompts=[
        "What is the capital of Ghana?",
        "Summarise the Ghana Investment Policy.",
        "What are the main sectors of Ghana's economy?",
    ],
    criticality="HIGH",
    label="ghana_agent_v2",
)

print(report.verdict)       # CONSISTENT if all pass
print(report.pass_rate)     # 1.0 = 100%
print(report.overall_score) # 0.93
```

### @consistency_probe Decorator

```python
from semantic_pass_k import consistency_probe

@consistency_probe(k=5, criticality="HIGH", raise_on_fail=True)
def answer_question(prompt: str) -> str:
    return call_my_llm(prompt)

answer = answer_question("What is 2 + 2?")
print(answer_question.last_consistency_result.verdict)
```

### pytest Integration

```python
from semantic_pass_k.pytest_plugin import assert_consistent

def test_agent_consistency():
    outputs = [my_agent("Summarise the report.") for _ in range(5)]
    assert_consistent(outputs, criticality="HIGH", agent_label="summariser")
```

---

## CLI

```bash
# Run consistency check on provided outputs
sempass run "Output A." "Output A." "Output A." --criticality HIGH

# JSON output for CI pipelines
sempass run "A" "B" "C" --criticality LOW --json-output

# Store results to SQLite
sempass run "A" "A" "A" --criticality HIGH --store-db history.db --label my_agent

# View history
sempass report --store-db history.db --label my_agent

# CI gate (exit 1 if INCONSISTENT)
sempass ci --store-db history.db --label my_agent

# Compare two agents
sempass budget --store-db history.db gpt-4o claude-3-5-sonnet
```

### GitHub Actions

```yaml
- name: Check agent consistency
  run: |
    sempass run "$OUTPUT1" "$OUTPUT2" "$OUTPUT3" "$OUTPUT4" "$OUTPUT5" \
      --criticality HIGH \
      --store-db consistency.db \
      --label production_agent
    sempass ci --store-db consistency.db --label production_agent
```

---

## Zero Dependencies

The default TF-IDF cosine backend requires **no external dependencies**.
All stdlib. No sentence-transformers, no numpy, no scikit-learn.

For higher-quality semantic embeddings, install the optional neural backend:
```bash
pip install "semantic-pass-k[neural]"
engine = ConsistencyEngine(use_neural=True)
```

---

## Why semantic-pass-k?

The math-based `pass@k` metric from HumanEval measures *exact match* — does at least one of k outputs pass a test case? That's for code generation.

`semantic-pass-k` measures *semantic equivalence* — do k outputs mean the same thing, even if worded differently? That's for agents.

### Prior Art

| Tool | What it does | What it misses |
|------|-------------|----------------|
| AgentAssay | How many runs do you need? | Doesn't produce ConsistencyScore |
| Promptfoo | LLM eval + comparison | No semantic pass@k metric |
| DeepEval | LLM eval metrics | No cross-run consistency gate |
| LangSmith | Tracing + eval | No criticality-tiered consistency |
| τ-bench | Research benchmark | Not pip-installable, not CI-gateable |

---

## Roadmap

- v0.2: FidelityDrift — track ConsistencyScore over time, alert on degradation
- v0.3: Hosted dashboard — visualise consistency trends across agent versions
- v0.4: ConsistencyBudget — auto-select k and criticality tier from task description

---

## Pattern Origin

Derived from PAT-062 (BibleWorld): Numbers 23:19 — *"Does he speak and then not act? Does he promise and then not fulfill?"*

Balaam's oracle encodes a three-part consistency verification protocol:
1. Declare the expected behavioral invariant
2. Run the empirical behavioral test
3. Verify the null discrepancy result

Consistency is not an internal property — it is verified externally by comparing outputs across time.

---

## License

MIT — use it, fork it, build on it.
