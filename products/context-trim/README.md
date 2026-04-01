# context-trim

> Production-grade conversational context window trimmer.
> Fit any conversation into your LLM token budget. Zero dependencies. No local model required.

[![PyPI](https://img.shields.io/pypi/v/context-trim)](https://pypi.org/project/context-trim/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Your conversation history is 12,000 tokens. Your model's context window is 8,192 tokens. You have two choices:

1. **Hard truncate** — silently drop old messages, lose context, confuse the model
2. **Give up** — throw a context length error and call it a bug

There's a third choice.

**context-trim** strategically trims conversation histories to fit within any token budget — preserving system messages, prioritising recent and important content, and giving you a full audit trail of what was dropped.

---

## Install

```bash
pip install context-trim
```

Zero hard dependencies. Works with any LLM provider (OpenAI, Anthropic, Gemini, local models).

---

## Quick Start

```python
from context_trim import ContextTrim, TrimStrategy

ct = ContextTrim(max_tokens=8192, reserved_tokens=512)

# Check if messages fit
if not ct.fits(messages):
    result = ct.trim(messages, strategy=TrimStrategy.HYBRID)
    messages = result.messages
    print(result.summary())
    # [context-trim] OK | strategy=hybrid | messages 45->12 | tokens 11200->3840 | dropped=33 | ratio=65.7%
```

---

## Five Trim Strategies

| Strategy | How it works | Best for |
|---|---|---|
| `RECENCY_FIRST` | Drop oldest messages first | Short-horizon chats, customer support |
| `IMPORTANCE` | Score each message, drop lowest first | Long research conversations |
| `SLIDING_WINDOW` | Keep the last N messages that fit | Fixed-window RAG contexts |
| `SUMMARY_POINTS` | Replace dropped messages with a bullet summary | When context history matters |
| `HYBRID` | Importance-ranked + recency bonus | **General purpose (default)** |

System messages (`"role": "system"`) are **always preserved** regardless of strategy.

---

## API Reference

### ContextTrim

```python
ct = ContextTrim(
    max_tokens=8192,        # total context window size
    reserved_tokens=512,    # tokens held back for system prompt + response
    db_path="history.db",   # optional: SQLite history (omit to skip)
)

ct.estimate(messages)       # → int: estimated token count
ct.fits(messages)           # → bool: does it fit within the budget?
ct.tokens_over(messages)    # → int: how many tokens over budget (0 if OK)
ct.trim(messages, strategy=TrimStrategy.HYBRID, pipeline_id="my-app")  # → TrimResult
ct.trim_document(text)      # → DocumentTrimResult: trim a long text
ct.ci_gate(messages)        # raises RuntimeError if over budget — use in CI
```

### TrimResult

```python
result = ct.trim(messages)

result.messages          # list[dict]: trimmed message list, ready to send
result.original_count    # int: messages before trimming
result.final_count       # int: messages after trimming
result.original_tokens   # int: estimated tokens before
result.final_tokens      # int: estimated tokens after
result.dropped_count     # int: messages dropped
result.trim_ratio        # float: fraction of tokens removed
result.within_budget     # bool: True if final fits in budget
result.summary()         # str: one-line human-readable summary
result.to_dict()         # dict: JSON-serialisable metadata
```

---

## CI Gate

Block builds that would send oversized context to your LLM:

```python
# In your test suite or CI script
from context_trim import ContextTrim

ct = ContextTrim(max_tokens=8192)
ct.ci_gate(messages)  # raises RuntimeError if over budget
```

CLI version:

```bash
context-trim ci messages.json --max-tokens 8192
# exit 0 if OK, exit 1 if over budget
```

---

## CLI

```bash
# Estimate token count
context-trim estimate messages.json --max-tokens 8192

# Trim and print to stdout
context-trim trim messages.json --max-tokens 8192 --strategy hybrid

# Trim and save result
context-trim trim messages.json --max-tokens 8192 --output trimmed.json

# CI gate
context-trim ci messages.json --max-tokens 8192

# Show history (requires --db-path when trimming)
context-trim history --db history.db --pipeline my-app
```

---

## History & Audit Trail

```python
from context_trim import ContextTrim, TrimStore

# Record every trim operation
ct = ContextTrim(max_tokens=8192, db_path="history.db")
result = ct.trim(messages, pipeline_id="my-rag-app")

# Query history
store = TrimStore("history.db")
print(store.stats("my-rag-app"))
# {'pipeline_id': 'my-rag-app', 'total_runs': 127, 'over_budget_runs': 0, 'avg_trim_ratio': 0.312}
```

---

## Token Estimation

`context-trim` uses a 4 chars/token heuristic — no tiktoken or sentencepiece required. This is accurate to ±10% for English content and ±20% for code/multilingual. For exact counts, wrap the trimmed output in your tokenizer before sending.

---

## Competitive Landscape

| Tool | What it does | Standalone? |
|---|---|---|
| LangChain ContextualCompression | RAG-focused, retrieval-time only | No (LangChain required) |
| LLMLingua (Microsoft) | Token-level compression via local LM | Requires local model |
| context-compressor-llm | Anchored summaries | v0.1.2, minimal adoption |
| **context-trim** | Strategy-based conversation trimming | **Yes — zero dependencies** |

`context-trim` is the only pip-installable library focused on **production conversation trimming** (not RAG retrieval, not model weight compression) with zero dependencies.

---

## License

MIT
