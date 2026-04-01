# context-trace

**Per-context-chunk causal AttributionScore for LLM outputs.**

> *"The wind blows wherever it pleases. You hear its sound, but you cannot tell where it comes from or where it is going."* — John 3:8

When your LLM produces output from a long context, which chunk caused it? `context-trace` answers this question with a named, CI-gateable **AttributionScore** per input chunk — no model internals required, works with any API (Claude, GPT-4o, Gemini, Llama).

---

## The problem

You're running a RAG pipeline or a complex agent with a 50K-token context. The LLM hallucinates a fact. Which retrieved document caused it? Which system instruction was ignored? You have no way to know — the output arrived, but its source is opaque.

Existing tools don't answer this:
- **Arize Phoenix / LangSmith / Langfuse**: trace execution flow, not causal attribution
- **Ragas**: ranks retrieved chunks by relevance, not by output influence
- **ContextCite (MadryLab)**: closest academic tool, but requires raw logit access — incompatible with any hosted API

## The solution

`context-trace` uses **counterfactual masking**: for each named context chunk, it runs the LLM `k` times with that chunk replaced by `[REMOVED]`, then measures how much the output changed (via embedding cosine similarity).

```
AttributionScore = 1.0 - mean(cosine_similarity(original_output, masked_output_i))
```

- Score **near 1.0** → removing this chunk dramatically changes the output → **high causal contribution**
- Score **near 0.0** → removing this chunk barely changes the output → **low causal contribution**

No logit access needed. Works with any LLM via a simple `Callable[[str], str]` runner interface.

---

## Install

```bash
pip install context-trace
```

For Anthropic API support:
```bash
pip install "context-trace[anthropic]"
```

---

## Quick start

```python
from context_trace import ContextTracer, AttributionGate, CostBudget
from context_trace.runners import anthropic_runner

runner = anthropic_runner(model="claude-haiku-4-5-20251001")

tracer = ContextTracer(runner=runner, k=3)

report = tracer.trace(
    prompt=full_prompt,         # the original prompt sent to the LLM
    original_output=response,   # the LLM's original response
    chunks={
        "system_prompt":   system_prompt_text,
        "retrieved_doc_1": doc1_text,
        "retrieved_doc_2": doc2_text,
        "tool_result":     tool_result_text,
        "user_message":    user_message_text,
    },
)

print(report.attribution_heatmap)
# system_prompt   [███░░░░░░░] 0.31
# retrieved_doc_1 [██████████] 0.87
# retrieved_doc_2 [███░░░░░░░] 0.24
# tool_result     [████████░░] 0.72
# user_message    [██░░░░░░░░] 0.18

print(report.top_contributors(n=3))
# [("retrieved_doc_1", 0.87), ("tool_result", 0.72), ("system_prompt", 0.31)]
```

---

## CI gate

```python
gate = AttributionGate(
    max_single_chunk_score=0.90,  # fail if any chunk dominates output (>90%)
    min_chunks_contributing=2,    # fail if fewer than 2 chunks contribute
)
gate.check(report)  # raises AttributionGateFailure if violated

# Non-raising variants
passed = gate.passed(report)
ok, violations = gate.result(report)
```

In pytest:
```python
def test_rag_does_not_hallucinate_from_single_doc(rag_pipeline):
    output = rag_pipeline.run("What is the capital of France?")
    report = tracer.trace(
        prompt=rag_pipeline.last_prompt,
        original_output=output,
        chunks=rag_pipeline.last_chunks,
    )
    gate = AttributionGate(max_single_chunk_score=0.85)
    assert gate.passed(report), f"Single chunk dominates: {report.attribution_heatmap}"
```

---

## CLI

```bash
# Run attribution from a config file
ctrace run --config ctrace.yaml --output report.json

# Display heatmap
ctrace show --report report.json --top 5

# CI gate check (exits 1 if violated)
ctrace gate --report report.json --max-score 0.90 --min-contributors 2

# Compare before/after (e.g., after adding a new document)
ctrace compare --baseline before.json --current after.json

# Estimate cost before running
ctrace estimate --config ctrace.yaml
```

`ctrace.yaml` format:
```yaml
runner:
  type: anthropic
  model: claude-haiku-4-5-20251001
  max_tokens: 512

chunks:
  system_prompt:
    source: prompts/system.txt
  retrieved_doc:
    inline: "Paris is the capital of France."
  user_message:
    inline: "What is the capital of France?"

prompt:
  source: prompts/full_prompt.txt
original_output:
  source: outputs/response.txt

k: 3

budget:
  max_api_calls: 100
  max_cost_usd: 0.50
```

---

## Cost control

Attribution requires `n_chunks × k` API calls per run. For 5 chunks with `k=3`: 15 calls ≈ $0.015 at Haiku pricing.

```python
from context_trace import CostBudget

budget = CostBudget(
    max_api_calls=50,     # hard cap: raises BudgetExceededError if exceeded
    max_cost_usd=0.10,
    cost_per_call_usd=0.001,
)
tracer = ContextTracer(runner=runner, k=3, budget=budget)
```

---

## SQLite store

```python
from context_trace import AttributionStore

store = AttributionStore("ctrace.db")
run_id = store.save(report, label="rag_v2_prod")

history = store.list_runs()   # [{id, label, created_at, top_score, ...}]
data    = store.get(run_id)   # full report dict
```

---

## pytest plugin

```python
# conftest.py
pytest_plugins = ["context_trace.pytest_plugin"]

@pytest.fixture
def ctrace_runner_fn():
    return my_llm_runner  # Callable[[str], str]

# test_rag.py
def test_attribution(ctrace_tracer):
    output = my_rag_pipeline("What is X?")
    report = ctrace_tracer.trace(prompt, output, chunks)
    assert report.top_score > 0.3
```

CLI options: `--ctrace-k N`, `--ctrace-budget-calls N`

---

## How it works

For each named chunk:

1. **Mask**: Replace chunk text in the original prompt with `[REMOVED:<chunk_name>]`
2. **Run k times**: Call the LLM runner on the masked prompt
3. **Embed**: Encode `original_output` and each masked output via `sentence-transformers`
4. **Score**: `attribution_score = 1.0 - mean(cosine_similarity(original, masked_i))`

Chunks where the text is not found verbatim in the prompt are **skipped with a warning** (not treated as errors).

---

## Testing without API calls

```python
from context_trace import ContextTracer
from context_trace.embedder import MockEmbedder

# Deterministic, no model download
tracer = ContextTracer(
    runner=my_mock_runner,
    embedder=MockEmbedder(dim=16),
    k=3,
)
```

---

## Competitive landscape

| Tool | What it does | API-compatible? | Named AttributionScore? |
|------|-------------|----------------|------------------------|
| **context-trace** | Per-chunk causal attribution | Yes (any runner) | Yes |
| ContextCite (MadryLab) | Chunk attribution via logit regression | No (needs logits) | No |
| Ragas | RAG evaluation metrics | Partial | No |
| LangSmith | Execution tracing | SaaS only | No |
| Arize Phoenix | ML observability | Yes | No |
| SHAP/LIME | ML feature attribution | No (tabular) | No |

---

## Why this exists

From John 3:8: *"The wind blows wherever it pleases... you cannot tell where it comes from."*
LLM outputs are like that wind — observable but causally opaque. `context-trace` is the instrument that tells you which part of the context was the wind.

---

## License

MIT. Built by [BuildWorld](https://github.com/buildworld-ai).
