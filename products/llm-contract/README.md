# llm-contract

**Define, version, and enforce behavioral contracts on LLM function calls.**

> Pydantic validates structure. llm-contract validates behavior. Together they make LLM function calls trustworthy.

[![PyPI version](https://badge.fury.io/py/llm-contract.svg)](https://badge.fury.io/py/llm-contract)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![CI](https://github.com/buildworld-ai/llm-contract/workflows/CI/badge.svg)](https://github.com/buildworld-ai/llm-contract/actions)

---

## The Problem

Your LLM functions have Pydantic validation. They don't have behavioral contracts.

Here's what that means in practice:

```python
# Pydantic catches this: ✓
{"title": 123, "summary": "..."}   # Wrong type — field validation fails

# Pydantic does NOT catch this: ✗
{"title": "Q3 Report", "summary": "Revenue was $12.4B (invented fact not in source doc)"}
# Structurally valid. Behaviorally wrong. Pydantic approves it. Your users see it.
```

This is the gap. And it costs teams — in bugs found by users, not tests.

**Real pain points this solves:**

- You switch from GPT-4o to Claude. Output structure looks correct. Behavior silently regressed.
- Your model provider updates their model. Your `summarize_document` function starts producing 8-bullet summaries instead of 3-5. No alert.
- You have 20+ LLM functions in production. You cannot watch all of them manually for drift.
- Your team defines "should produce X" in comments. No enforcement. No versioning. No shared standard.

**llm-contract fixes all of this.**

---

## Quick Start

```bash
pip install llm-contract[anthropic]
```

```python
from pydantic import BaseModel
from typing import List
from llm_contract import contract, SemanticRule
import anthropic

class DocumentSummary(BaseModel):
    title: str
    summary: str
    key_points: List[str]

@contract(
    schema=DocumentSummary,
    semantic_rules=[
        SemanticRule(
            name="no_fabrication",
            description="Summary must not introduce facts not present in the source document",
            weight=1.0,  # Critical — contract fails if violated
        ),
        SemanticRule(
            name="key_points_count",
            description="Must include 3-5 key points, no more, no less",
            weight=0.8,
        ),
    ],
    version="1.0.0",
    on_violation="raise",
)
def summarize_document(document: str) -> DocumentSummary:
    client = anthropic.Anthropic()
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"Summarize as JSON: {document}"}]
    )
    import json
    return DocumentSummary(**json.loads(response.content[0].text))

# Call it — violations raise ContractViolationError
result = summarize_document(my_document)
print(result.summary)  # Guaranteed to meet your behavioral contract
```

That's it. Three steps. Your LLM function now has a behavioral contract.

---

## Features

### Two-Layer Validation

```python
@contract(
    schema=MyOutputSchema,      # Layer 1: Pydantic structural validation (0ms)
    semantic_rules=[...],       # Layer 2: LLM-judge behavioral validation (~200ms)
    version="2.0.0",
)
```

**Layer 1** catches: wrong field types, missing required fields, invalid enum values. Zero latency — Pydantic is fast.

**Layer 2** catches: fabricated facts, wrong tone, missing required content, semantic inconsistency. Configurable LLM judge.

---

### Contract Versioning (SemVer for Behavior)

```python
@contract(schema=SummarySchema, version="2.1.0")
def summarize_document(doc: str) -> SummarySchema: ...
```

Behavioral versioning follows semantic versioning rules:
- **Major** (`1.x.x` → `2.0.0`): Breaking behavioral change
- **Minor** (`x.1.x` → `x.2.0`): New behavioral requirement, backward compatible
- **Patch** (`x.x.1` → `x.x.2`): Threshold adjustment, no behavioral change

Version is attached to the wrapper for introspection:
```python
summarize_document.__contract_version__  # "2.1.0"
```

---

### Provider-Agnostic Enforcement

Same contract, any provider:

```python
pip install llm-contract[anthropic]   # Claude judge
pip install llm-contract[openai]      # GPT judge
pip install llm-contract[all]         # Both
```

```python
@contract(schema=SummarySchema, version="1.0.0")
def summarize_anthropic(doc: str) -> SummarySchema:
    # Your Claude-based implementation
    ...

@contract(schema=SummarySchema, version="1.0.0")
def summarize_openai(doc: str) -> SummarySchema:
    # Your OpenAI-based implementation
    ...
```

Switching providers doesn't break the contract.

---

### Violation Handling Strategies

```python
@contract(..., on_violation="raise")    # Default — raise ContractViolationError
@contract(..., on_violation="warn")     # Log warning, return output anyway
@contract(..., on_violation="log")      # Silently log to SQLite
@contract(..., on_violation="fallback", fallback=my_fallback_fn)  # Call fallback
```

Catch violations gracefully:

```python
from llm_contract import ContractViolationError

try:
    result = summarize_document(doc)
except ContractViolationError as e:
    print(f"Failed rules: {[r.rule_name for r in e.result.failed_rules]}")
    print(f"Overall score: {e.result.overall_score:.2%}")
```

---

### Drift Detection (SQLite-backed)

Every contract evaluation is logged to a local SQLite database. You own your data.

```bash
# Check pass rates across all contracts
llm-contract validate --min-pass-rate 0.90

# Output:
# ✓ summarize_document v1.0.0 — 94.0% (PASS) [threshold: 90%] [50 evals]
# ✗ generate_report v1.0.0   — 76.0% (FAIL) [threshold: 90%] [50 evals]
# GATE FAIL — one or more contracts below threshold.

# Detect behavioral drift over 30 days
llm-contract drift-report --last 30d

# Output:
# ! summarize_document v1.0.0 | 95.2% → 87.1% (-8.1pp) | DRIFT DETECTED [100 evals]
# ✓ extract_entities v2.1.0  | 98.0% → 99.0% (+1.0pp) | stable [80 evals]
```

---

### CI/CD Gate

Use the CLI as a CI gate after any model or provider change:

```yaml
# .github/workflows/llm-validate.yml
- name: Validate LLM contracts
  run: llm-contract validate --min-pass-rate 0.90 --days 7
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## How It Works

```
Your LLM function call
        │
        ▼
@contract decorator intercepts return value
        │
        ├── Layer 1: Pydantic structural validation (0ms)
        │       └── Field names, types, required fields → pass/fail
        │
        ├── Layer 2: SemanticRule evaluation (opt-in, ~200ms/rule)
        │       └── For each enabled rule: calls LLM judge
        │       └── Aggregates weighted pass/fail scores
        │       └── Critical rules (weight=1.0): one failure = contract fails
        │
        ├── Drift logger
        │       └── Writes result to SQLite (timestamp, provider, model, pass/fail)
        │
        └── Violation handler
                └── raise | warn | log | fallback
```

The LLM judge (default: `claude-haiku-4-5-20251001`) is called once per SemanticRule.

Semantic validation is opt-in and can be disabled in performance-critical paths:
```python
@contract(schema=MySchema, version="1.0.0", validate_semantic=False)
```

---

## Configuration

```python
import llm_contract

llm_contract.configure(
    default_judge_model="claude-haiku-4-5-20251001",
    default_judge_provider="anthropic",
    db_path="./llm_contract_logs.db",
    log_all_results=True,
    default_threshold=0.90,
)
```

Environment variables:
```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
LLM_CONTRACT_DB_PATH=./logs/contracts.db
LLM_CONTRACT_JUDGE_MODEL=claude-haiku-4-5-20251001
LLM_CONTRACT_JUDGE_PROVIDER=anthropic
```

---

## Installation

```bash
# With Anthropic judge (recommended)
pip install llm-contract[anthropic]

# With OpenAI judge
pip install llm-contract[openai]

# Both providers + CLI
pip install llm-contract[all]

# Core only (no LLM judge — structural validation only)
pip install llm-contract
```

---

## Comparison

| | llm-contract | Pydantic | DeepEval | Promptfoo* |
|--|--|--|--|--|
| Structural validation | ✓ | ✓ | ✗ | ✗ |
| Behavioral contracts | ✓ | ✗ | Partial | ✗ |
| Contract versioning | ✓ | ✗ | ✗ | ✗ |
| Runtime enforcement | ✓ | ✓ | ✗ | ✗ |
| Drift detection | ✓ | ✗ | ✗ | ✗ |
| CI gate | ✓ | ✗ | ✓ | ✓ |
| Provider-agnostic | ✓ | N/A | ✓ | ✓ |
| `pip install` | ✓ | ✓ | ✓ | ✗ (npm) |
| Self-hosted data | ✓ | N/A | ✗ | ✗ |

*Promptfoo acquired by OpenAI (March 2026) — provider lock-in concern for existing users.

llm-contract works best alongside Pydantic (structure), DeepEval (quality benchmarking), and Langfuse (observability). It fills the behavioral contract gap that none of them address.

---

## Roadmap

**v0.1 (current)**
- `@contract` decorator with structural + semantic validation
- `ContractViolationError` with full violation details
- SQLite drift logging — self-hosted, zero dependencies
- `llm-contract validate` CLI gate
- `llm-contract drift-report` CLI
- Claude and OpenAI judge support

**v0.2**
- GitHub Action: `llm-contract/action@v1`
- Contract registry (local + team-shared)
- `llm-contract compare --before COMMIT --after COMMIT`
- pytest plugin (`ContractSuite`)

**v0.3**
- Hosted dashboard (bring-your-own SQLite)
- Slack / PagerDuty drift alerts
- Multi-model provider comparison mode
- Contract inheritance

---

## Contributing

We welcome contributions. Key areas:

- Additional LLM judge implementations (Mistral, local models via Ollama)
- More SemanticRule templates (common patterns for summarization, extraction, classification)
- Performance optimization for high-throughput production use

```bash
git clone https://github.com/buildworld-ai/llm-contract
cd llm-contract/products/llm-contract
pip install -e ".[dev]"
pytest tests/
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

*Built by engineers who got paged one too many times because an LLM function changed its behavior after a model update.*
