# prompt-lock

**Git-native prompt regression testing with judge calibration.**

[![PyPI version](https://badge.fury.io/py/prompt-lock.svg)](https://pypi.org/project/prompt-lock/)
[![CI](https://github.com/buildworld-ai/prompt-lock/actions/workflows/ci.yml/badge.svg)](https://github.com/buildworld-ai/prompt-lock/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Guards at the gaps in your LLM CI/CD pipeline. Fails the build when a prompt change causes a regression — and verifies that your LLM judge actually agrees with humans before trusting it as a gate.

```
pip install prompt-lock
```

---

## The problem

You changed a prompt. Did your model outputs get worse?

You probably don't know. 82% of teams have no automated detection for prompt quality regressions. The few that do often use LLM-as-a-judge — but their judge is miscalibrated: it disagrees with human evaluators on 20–40% of examples and they've never measured it.

## The solution

prompt-lock does three things no other tool does together in a single `pip install`:

1. **Detects changed prompts via git diff** — only evaluates what changed, keeping costs low
2. **Verifies judge calibration** — runs your LLM judge against human-labeled examples, measures agreement rate and Spearman correlation, and *blocks the CI pipeline if the judge can't be trusted*
3. **Regression gate** — fails the build if eval scores drop more than a configurable threshold from baseline

---

## Quick start

```bash
pip install prompt-lock
cd your-llm-project
prompt-lock init
```

`init` creates `.prompt-lock.yml`, `prompts/`, `tests/test_cases.jsonl`, and `tests/human_labels.jsonl`.

Fill in your test cases:
```jsonl
{"input": "Summarize this article: ...", "output": "The article discusses ...", "expected_output": "A summary of the article."}
```

Run:
```bash
prompt-lock check
```

---

## Configuration

```yaml
# .prompt-lock.yml
version: "1"
model: gpt-4o-mini

# Judge calibration — the key differentiator
judge:
  enabled: true
  human_labels_file: tests/human_labels.jsonl
  model: gpt-4o-mini
  criteria: "Rate the quality of this response from 0.0 to 1.0."
  min_agreement: 0.80   # 80% of examples must agree (within ±0.15)
  min_spearman: 0.70    # Spearman correlation with human scores

prompts:
  - path: "prompts/*.txt"
    name: "My Prompts"
    test_cases_file: tests/test_cases.jsonl
    evals:
      - type: llm_judge
        criteria: "Is the response helpful, accurate, and well-structured?"
        threshold: 0.70
      - type: semantic_similarity
        threshold: 0.80

gate:
  mode: regression      # hard | regression | soft
  regression_threshold: 0.05   # fail if score drops >5% from baseline
```

---

## Eval types

| Type | What it checks | Requires |
|------|---------------|----------|
| `llm_judge` | LLM scores output against criteria (0.0–1.0) | `criteria` |
| `semantic_similarity` | Cosine similarity to expected output (offline, all-MiniLM-L6-v2) | `expected_output` in test cases |
| `exact_match` | Exact string match | `expected_output` in test cases |
| `regex` | Output matches a regex pattern | `pattern` |
| `custom` | Your own Python function `fn(input, output) -> float` | `custom_fn` |

Works with any LLM provider via [LiteLLM](https://github.com/BerriAI/litellm):
- `gpt-4o-mini`, `gpt-4o`
- `claude-haiku-4-5-20251001`, `claude-sonnet-4-6`
- `mistral/mistral-small`
- Any local model via Ollama: `ollama/llama3`

---

## Gate modes

**`regression`** (default) — fail if score drops more than `regression_threshold` from recent baseline. Good for ongoing development.

**`hard`** — fail if score is below `hard_threshold`. Good for critical prompts with known minimum quality.

**`soft`** — never fail, warn only. Good for new prompts without established baselines.

---

## Judge calibration

The unique feature. Before running evals, prompt-lock checks whether your LLM judge actually agrees with human evaluators:

```bash
prompt-lock calibrate
```

```
┌─────────────────────────────────────────────────────────────┐
│ Calibration Summary                                         │
│                                                             │
│ PASSED                                                      │
│                                                             │
│ Agreement rate   87.5%  (min: 80%)                         │
│ Spearman r       0.831  (min: 0.70)                         │
│ Bias             +0.042  (positive = judge inflates scores) │
│ Examples         16                                         │
└─────────────────────────────────────────────────────────────┘
```

If calibration fails, `prompt-lock check` exits with code 2 and blocks deployment. Your CI pipeline doesn't trust an uncalibrated judge.

Create `tests/human_labels.jsonl`:
```jsonl
{"input": "What is 2+2?", "output": "The answer is 4.", "human_score": 1.0}
{"input": "What is 2+2?", "output": "It's roughly 5.", "human_score": 0.0}
{"input": "Explain Python.", "output": "Python is a high-level language.", "human_score": 0.9}
```
Minimum 5 examples. More is better.

---

## GitHub Actions

```yaml
# .github/workflows/prompt-lock.yml
name: Prompt Regression Tests

on: [push, pull_request]

jobs:
  prompt-lock:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 2   # needed for git diff detection

      - uses: buildworld-ai/prompt-lock@v1
        with:
          config: .prompt-lock.yml
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Or with other providers:
```yaml
env:
  ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
```

---

## CLI reference

```bash
prompt-lock init                    # initialize config and example files
prompt-lock check                   # run regression checks (git-diff aware)
prompt-lock check --all-prompts     # eval all prompts, not just changed ones
prompt-lock check --no-calibrate    # skip calibration check
prompt-lock check -v                # verbose: show per-test-case results
prompt-lock calibrate               # run calibration and show detailed results
prompt-lock traces show             # show recent eval runs from trace ledger
prompt-lock traces show -n 50       # show last 50 runs
prompt-lock traces diff abc123 def456  # compare scores between two commits
```

---

## Trace ledger

Every eval run is recorded in a local SQLite database (`.prompt-lock/traces.db`) with the git commit SHA. This is how regression detection works — it compares current scores to recent passing baselines.

```bash
prompt-lock traces show

┌───────────────────────┬─────────┬─────────────────┬───────────┬───────┬──────┐
│ Timestamp             │ Commit  │ Prompt          │ Type      │ Score │ Pass │
├───────────────────────┼─────────┼─────────────────┼───────────┼───────┼──────┤
│ 2026-03-27T14:32:01   │ a1b2c3d │ prompts/sum.txt │ llm_judge │ 0.841 │ ✓    │
│ 2026-03-27T14:32:00   │ a1b2c3d │ prompts/sum.txt │ semantic  │ 0.923 │ ✓    │
│ 2026-03-26T09:15:44   │ e4f5g6h │ prompts/sum.txt │ llm_judge │ 0.710 │ ✓    │
└───────────────────────┴─────────┴─────────────────┴───────────┴───────┴──────┘
```

---

## Why not Promptfoo / LangSmith / DeepEval?

| Capability | prompt-lock | Promptfoo | LangSmith | DeepEval |
|-----------|:-----------:|:---------:|:---------:|:--------:|
| Git-diff aware (only eval changed prompts) | ✓ | ✗ | ✗ | ✗ |
| Judge calibration against human labels | ✓ | ✗ | partial | ✗ |
| Block CI if judge is miscalibrated | ✓ | ✗ | ✗ | ✗ |
| Regression gate (baseline comparison) | ✓ | ✓ | ✓ | ✓ |
| Commit-linked trace ledger | ✓ | ✗ | ✓ | ✗ |
| Framework-agnostic (LiteLLM) | ✓ | ✓ | ✗ | ✓ |
| Offline semantic similarity | ✓ | ✗ | ✗ | ✓ |
| Zero hosted infrastructure | ✓ | ✓ | ✗ | partial |
| `pip install` in 30 seconds | ✓ | ✗ | ✗ | ✓ |

Promptfoo was acquired by OpenAI in March 2026 — its roadmap is now OpenAI-aligned. prompt-lock is MIT licensed and provider-agnostic.

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## License

MIT. Built by [BuildWorld](https://github.com/buildworld-ai).

*Guards at the gaps. Nehemiah 4:13.*
