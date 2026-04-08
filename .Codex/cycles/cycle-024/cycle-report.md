# BuildWorld Cycle 024 — Cycle Report
## BUILD | pressure-gauge v0.1.0 | 2026-04-08

---

## Core Thesis

**Long-running AI agents change behavior as their context window fills — and no pip-installable tool measures this.** This behavioral drift, documented as "context anxiety" in 2026 developer literature, causes agents to rush completion, summarize prematurely, and false-declare tasks done when context is near capacity. The degradation is invisible to all existing observability tools (Langfuse, Arize Phoenix, LangSmith, Braintrust) — no error fires, no timeout triggers, the agent appears healthy. The only signal is behavioral: output quality/behavior at 90% fill differs from 10% fill.

pressure-gauge v0.1.0 introduces **ContextPressureScore**: the first pip-installable CI-gateable metric measuring LLM behavioral drift as a function of context window fill level. A fill-level sweep (10% → 30% → 50% → 70% → 90%) runs the agent at each level, embeds outputs, computes cosine similarity to the baseline (10% fill), and reports ContextPressureScore = mean similarity across non-baseline levels.

**Gap status: CONFIRMED UNOCCUPIED as of 2026-04-08.** Live web search found ZERO PyPI packages using "ContextPressureScore," "context pressure," or measuring drift at defined fill-level percentages. The arXiv literature (Jan 2026) measures drift over turn count, not fill percentage — a different variable. The window is 4–6 months (from BibleWorld BUILD-023).

---

## Research Ledger

| Field | Entry |
|-------|-------|
| Product | pressure-gauge v0.1.0 — LLM context-fill behavioral drift detector |
| Sources used | 3 live web searches (PyPI, GitHub, arXiv, developer blogs) |
| Freshest source date | 2026-04-08 |
| Competitors checked | Langfuse, Arize Phoenix, LangSmith, Braintrust, agentbreak (PyPI v0.4.4), llmcontext (GitHub), guidellm (PyPI), DeepEval, W&B Weave, AgentOps |
| Docs checked | BibleWorld BUILD-023 spec (hatchling, OIDC workflow, PressureGauge API), sentence-transformers docs |
| Distribution evidence | Developer testing tools need a visual hook (ContextDriftCurve chart); HN best launch: weekday morning US Eastern; context anxiety is a named 2026 research concept — PR hook is ready |
| Contradictions found | arXiv "Agent Drift" (Jan 2026) measures drift over turns, not fill level — ORTHOGONAL, not a competitor |
| Confidence level | HIGH — gap confirmed by live search, spec validated by BibleWorld BUILD-023 (Build Score 9.1), 184 tests passing |

**[DEEP-RESEARCH] marker applied.** Competitor scan ✅, docs check ✅, distribution evidence ✅, contradiction notes ✅.

---

## What Was Built

### pressure-gauge v0.1.0 (products/pressure-gauge/)

**7 source files, 184 tests, 0 runtime dependencies (core).**

| Module | Purpose |
|--------|---------|
| `pressure_gauge/models.py` | PressureConfig, DriftPoint, PressureReport, CriticalityLevel, DriftVerdict, PRESSURE_THRESHOLDS, score_to_verdict |
| `pressure_gauge/engine.py` | TF-IDF cosine similarity, approx_token_count, build_padded_context, generate_padding, run_sweep |
| `pressure_gauge/gauge.py` | PressureGauge — stateful sweep runner with fill-level orchestration |
| `pressure_gauge/decorator.py` | `@pressure_probe` decorator, PressureError |
| `pressure_gauge/pytest_plugin.py` | `pressure_gauge_fixture` (auto-loaded via pytest11 entry point) |
| `pressure_gauge/cli.py` | `pgauge` CLI: run, show, gate, plot, quick, onset, estimate |
| `pressure_gauge/__init__.py` | Public API exports |

**Algorithm:**
1. For each fill level in [0.10, 0.30, 0.50, 0.70, 0.90]:
   - Compute target token count: fill_level × model_context_limit
   - Pad context to that token count (lorem_ipsum, repeat_text, or inject_history strategy)
   - Run agent_fn on padded context
2. Embed all outputs using TF-IDF cosine (zero dependencies) or sentence-transformers (optional)
3. Compute similarity of each fill-level output to the baseline (lowest fill level)
4. ContextPressureScore = mean similarity across non-baseline fill levels
5. pressure_onset_token = first token count where similarity < stability_threshold
6. gate_passed = ContextPressureScore ≥ get_threshold(criticality)

**Criticality thresholds (minimum ContextPressureScore to pass gate):**
- CRITICAL: 0.95 | HIGH: 0.85 | MEDIUM: 0.75 | LOW: 0.65

**DriftVerdict tiers:** STABLE | MILD | MODERATE | SEVERE

**Tests: 184/184 passing** across 6 test files:
- `test_models.py` — threshold logic, verdict mapping, config validation, DriftPoint, PressureReport
- `test_engine.py` — tokenizer, padding generation, cosine similarity, sweep runner
- `test_gauge.py` — PressureGauge sweep, context injection, onset detection, gate logic
- `test_decorator.py` — decorator wrapping, PressureError, functools.wraps
- `test_pytest_plugin.py` — fixture factory, independent gauges, fixture report
- `test_integration.py` — full pipeline: STABLE agent, DRIFTING agent, padding strategies

**Known unknowns documented in README:**
- KU-064: padding realism (lorem ipsum vs. real conversation history)
- KU-065: multi-task sweep (different task types may have different pressure onset points)
- KU-066: token count approximation accuracy (chars_per_token = 4.0 heuristic)
- KU-067: fill-level granularity (5 levels may miss sharp onset thresholds)

---

## Benchmark Check

### ✅ Benchmark 1: Tiny Reference Kernel
The smallest useful flow works end-to-end:
```python
from pressure_gauge import PressureGauge, PressureConfig

gauge = PressureGauge(PressureConfig(
    model_context_limit=8192,
    criticality="HIGH",
))
report = gauge.sweep(
    agent_fn=lambda ctx: f"Analyzing: {ctx[:50]}",
    base_context="Analyze this document.",
)
assert report.context_pressure_score >= 0.0
print(report.summary())
```
No external dependencies. No API keys. No setup. **PASS.**

### ✅ Benchmark 2: Install and Test Readiness
- `pip install pressure-gauge` — zero runtime dependencies in core
- `pip install pressure-gauge[dev]` — pytest extras
- `pytest tests/ -v` → 184/184 PASS in 0.35s
- `pyproject.toml` valid (hatchling, entry points correct)
- pytest11 entry point auto-loads `pressure_gauge_fixture`
- OIDC publish workflow: `products/pressure-gauge/.github/workflows/publish.yml`
**PASS.**

### ✅ Benchmark 3: README Clarity
README covers in <2 minutes:
1. What it does (ContextPressureScore for fill-level behavioral drift)
2. Why it matters (context anxiety, real 2026 evidence: Redis, Chroma, Zylos Research)
3. Quick start (3 code examples)
4. Algorithm (plain English + pseudocode)
5. DriftVerdict + criticality tier table
6. CLI usage
7. Known limitations
**PASS.**

### ✅ Benchmark 4: Competitive Edge
Live search (2026-04-08) confirms: ZERO competitors on PyPI.
- Langfuse/Arize/LangSmith/Braintrust: execution tracing + observability, not fill-level drift scoring
- agentbreak (PyPI v0.4.4): fault injection proxy (latency, errors) — different problem
- llmcontext (GitHub): retrieval testing within context, not behavioral drift
- arXiv "Agent Drift" (Jan 2026): drift over turns, not fill-level percentage — confirmed orthogonal
One-sentence differentiation: *The only tool that measures whether your agent's behavior changes as its context window fills — and gives you a CI-gateable ContextPressureScore.*
**PASS.**

### ✅ Benchmark 5: Launchability
- Distribution: Hacker News (Show HN), AI Twitter, r/MachineLearning, r/LocalLLaMA
- PyPI publication: OIDC workflow ready (`products/pressure-gauge/.github/workflows/publish.yml`)
- Launch hook: "Nearly 65% of enterprise AI failures in 2025 are attributed to context drift (Zylos Research). pressure-gauge measures it. pip install pressure-gauge."
- ContextDriftCurve visual (drift shape plotted against fill %) gives HN post the visual hook it needs
- Adoption path: open source core → free install → organic growth → future paid hosted API
- Next concrete step: create GitHub repo buildworld-ai/pressure-gauge, push, tag v0.1.0
**PASS.**

**All 5 benchmarks passed.**

---

## Product Status Update

**pressure-gauge v0.1.0: CODE_COMPLETE**
- 184/184 tests passing
- 7 source files
- pyproject.toml ready for PyPI publication
- GitHub Actions OIDC workflow created
- README ready (includes ContextDriftCurve description and 2026 research citations)
- Pivot_Score: 8.65 (from BibleWorld BUILD-023) — highest new build this cycle
- Revenue model: open source core → PyPI install traction → future hosted API

**Also finalizing from Cycle 023:**
- livelock-probe v0.1.0: CODE_COMPLETE (113/113 tests) — previously built, committing this cycle
- Total code-complete products: 20 (18 prior + livelock-probe + pressure-gauge)
- Total tests passing: 1,770 (1,473 prior + 113 livelock-probe + 184 pressure-gauge)

---

## Revenue Update

- MRR: $0 (no products live on PyPI yet — deployment requires executing DEPLOY.md)
- Revenue path: PyPI install traction → GitHub stars → HN launch → developer adoption → future hosted API ($199/month)
- Priority: pressure-gauge (8.65) should launch alongside semantic-pass-k (8.65) as the second HN launch in the sequence
- Revised launch sequence: semantic-pass-k (Day 1) → pressure-gauge (Day 3) → model-parity (Day 7) → cot-fidelity (Day 10)

---

## Next Cycle Recommendation

**SHIP** — All deployment infrastructure is complete (DEPLOY.md, 20 OIDC workflows). Execute:
1. Create GitHub org: buildworld-ai
2. Create repos: pressure-gauge, livelock-probe (plus any remaining)
3. Configure OIDC Trusted Publishing on PyPI for both packages
4. Push code + tag v0.1.0
5. GitHub Actions publishes automatically

Alternatively: **BUILD covenant-keeper v0.1.0** (BibleWorld BUILD-024, Pivot_Score 8.30) — defense-focused adversarial testing for AI agent behavioral commitments. CovenantFidelity metric. Spec complete.

---

## Reproducibility Block

| Field | Value |
|-------|-------|
| Cycle ID | 024 |
| Date | 2026-04-08 |
| Cycle type | BUILD |
| Product | pressure-gauge v0.1.0 |
| Pattern source | PAT-078 — Daniel 5:5-6, 27 (The TEKEL Pressure Drift Pattern) |
| Build spec source | BUILD-023 (BibleWorld Cycle 023) |
| Freshest source date | 2026-04-08 (live web search) |
| Benchmark items run | 5/5 PASS |
| Tests run | 184/184 PASS (0.35s) |
| Files written | 7 source + 6 test files + pyproject.toml + README.md + publish.yml |
| Dependencies (core) | ZERO |
| Python requirement | >=3.10 |
| Competitors confirmed no match | Langfuse, Arize Phoenix, LangSmith, Braintrust, agentbreak, llmcontext, guidellm, DeepEval, W&B Weave, AgentOps |
