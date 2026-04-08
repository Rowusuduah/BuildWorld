# livelock-probe

**Detect when an AI agent is making zero net progress toward its goal.**

The only tool that measures structural livelock in AI agents: active, not erroring, going nowhere.

[![PyPI version](https://img.shields.io/pypi/v/livelock-probe.svg)](https://pypi.org/project/livelock-probe/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://pypi.org/project/livelock-probe/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## The Problem

AI agents deployed in production enter **structurally stuck states** where they are actively working but making zero net progress. These livelock states are invisible to standard monitoring:

- No error is thrown
- No timeout fires
- Spans complete "successfully"
- The agent appears healthy in every observability dashboard

The only signal: **token budget consumed without task completion.**

**Concrete instances:**

| Pattern | What happens |
|---------|-------------|
| RAG retrieval livelock | Agent retrieves results below quality threshold. Retries. Gets similar results. Retries indefinitely. Every HTTP 200. No error. |
| Evaluation loop livelock | Agent self-evaluates output. Output passes some criteria, fails others. Revises. New output passes different criteria, fails others. Loop until context exhaustion. |
| Multi-agent coordination livelock | Agent A waits for Agent B's output. Agent B waits for Agent A's confirmation. Neither proceeds. |
| Tool rate-limit livelock | Agent retries rate-limited tool with exponential backoff. Higher-priority requests always win. Agent retries forever. |
| Claude Code quota exhaustion | Documented by Anthropic (March 2026): "people are hitting usage limits way faster than expected." Primary cause: retry loops in livelock. |

## What existing tools miss

| Tool | Detects Livelock | LivelockScore | Progress Vector | CI Gate | Framework-Agnostic |
|------|:---:|:---:|:---:|:---:|:---:|
| **livelock-probe** | ✅ | ✅ | ✅ | ✅ | ✅ |
| Langfuse | ❌ | ❌ | ❌ | ❌ | ✅ |
| Arize Phoenix | ❌ | ❌ | ❌ | ❌ | ✅ |
| AgentRx (Microsoft) | ❌* | ❌ | ❌ | ❌ | ❌ |
| Braintrust | ❌ | ❌ | ❌ | ❌ | ✅ |
| LangSmith | ❌ | ❌ | ❌ | ❌ | ❌ |

*AgentRx detects the first unrecoverable step — a different problem. Livelock steps look recoverable.

---

## Installation

```bash
# Core (zero dependencies)
pip install livelock-probe

# With CLI (click + rich)
pip install livelock-probe[cli]

# With neural embeddings (sentence-transformers)
pip install livelock-probe[neural]

# Everything
pip install livelock-probe[full]
```

---

## Quick start

### Manual instrumentation

```python
from livelock_probe import LivelockSuite, ProgressConfig

suite = LivelockSuite(ProgressConfig(
    goal="resolve the database connection error",
    k=5,            # 5 consecutive stuck steps = livelock
    criticality="HIGH",
))

# Run your agent loop
for step in range(max_steps):
    output = agent.run_step(context)
    suite.record_step(output)

    # Early exit if livelock detected
    if suite.step_count() >= suite._config.k and not suite.gate():
        print("Livelock detected — stopping agent")
        break

report = suite.compute()
print(report.summary())
# [LIVELOCK_DETECTED] livelock_score=0.800 threshold=0.15 steps=10 max_consecutive_stuck=8/5
```

### Decorator

```python
from livelock_probe import livelock_probe_decorator

@livelock_probe_decorator(
    goal="complete the customer support ticket",
    k=5,
    criticality="HIGH",
    raise_on_livelock=True,  # raises LivelockError when budget exceeded in livelock
)
def agent_step(ticket):
    return llm.call(ticket)

for i in range(max_steps):
    output = agent_step(ticket)
    # Check gate every N steps
    if i % 5 == 0 and not agent_step._livelock_suite.gate():
        break

report = agent_step._livelock_suite.compute()
```

### pytest fixture

```python
def test_support_agent_no_livelock(livelock_suite):
    suite = livelock_suite(
        goal="resolve customer support ticket",
        k=5,
        criticality="HIGH",
    )
    steps = run_support_agent(sample_ticket)  # returns list of step outputs
    suite.record_steps(steps)
    report = suite.compute()
    assert report.gate_passed, (
        f"Agent livelock: LivelockScore={report.livelock_score:.3f}, "
        f"stuck from step {report.stuck_window_start}"
    )
```

### Context manager

```python
from livelock_probe import LivelockSuite, ProgressConfig

suite = LivelockSuite(ProgressConfig(goal="accomplish the task", k=4))

with suite.monitor() as m:
    for _ in range(max_steps):
        output = agent.run()
        m.record(output)

report = suite.compute()
```

---

## The Algorithm: LivelockScore

```
1. For each agent step output, compute cosine similarity to the goal description.
2. Build progress_vector: [sim(step_0, goal), sim(step_1, goal), ...]
3. Build progress_deltas:
       delta[0] = progress_vector[0]
       delta[i] = progress_vector[i] - progress_vector[i-1]  for i > 0
4. A step is "stuck" if |delta[i]| < epsilon.
5. LivelockScore = (number of stuck steps) / (total steps)
6. livelock_detected = (max consecutive stuck steps) >= k
```

**LivelockScore** is a scalar in [0.0, 1.0]:
- `0.0` = all steps progressing toward the goal
- `1.0` = all steps stuck (zero net progress per step)

Default similarity: smoothed TF-IDF cosine (**zero dependencies**).
Neural option: sentence-transformers `all-MiniLM-L6-v2` (`pip install livelock-probe[neural]`).

---

## Criticality Tiers

| Tier | Max LivelockScore | Use case |
|------|:-----------------:|---------|
| `CRITICAL` | 5% | Medical, legal, financial outputs |
| `HIGH` | 15% | Production agents, customer-facing |
| `MEDIUM` | 30% | Internal tools, best-effort tasks |
| `LOW` | 50% | Exploratory / brainstorming agents |

---

## Configuration

```python
from livelock_probe import ProgressConfig

config = ProgressConfig(
    goal="resolve the user's database connection issue",  # required
    k=5,              # consecutive stuck steps before livelock_detected=True
    epsilon=0.05,     # |progress_delta| < epsilon → step is stuck
    criticality="HIGH",
    budget_steps=100, # soft step limit (for external orchestration)
    use_neural=False, # True = sentence-transformers (requires [neural] extra)
    similarity_fn=None,  # injectable: (str, str) -> float
    agent_label="support_agent",
)
```

---

## LivelockReport fields

```python
report.livelock_score        # float [0.0, 1.0]
report.livelock_detected     # bool
report.verdict               # "LIVELOCK_FREE" | "BORDERLINE" | "LIVELOCK_DETECTED"
report.gate_passed           # bool (True = safe)
report.stuck_window_start    # int or None (step where longest stuck run begins)
report.stuck_window_end      # int or None
report.max_consecutive_stuck # int
report.progress_vector       # list[float] — similarity to goal per step
report.progress_deltas       # list[float] — per-step progress change
report.mean_progress         # float
report.recommendation        # str — human-readable guidance
report.summary()             # str — one-line summary
report.to_dict()             # dict — JSON-serialisable
```

---

## CLI

```bash
# Install CLI extras
pip install livelock-probe[cli]

# Demonstration (no dependencies required)
lprobe demo

# Estimate parameters
lprobe estimate --k 5 --epsilon 0.05 --criticality HIGH

# Generate a report from a saved trace
# trace.json: {"steps": ["output 1", "output 2", ...]}
lprobe report trace.json --goal "resolve DB error" --format json -o report.json

# Gate check in CI (exit 1 if livelock)
lprobe gate report.json

# Pretty-print a saved report
lprobe show report.json
```

---

## CI/CD integration

```yaml
# .github/workflows/agent-test.yml
- name: Run agent livelock tests
  run: pytest tests/ -v

# Gate check from saved trace
- name: Livelock gate
  run: lprobe gate livelock-report.json
```

---

## Known limitations (v0.1.0)

- **KU-060:** Default `epsilon=0.05` is a calibrated hypothesis. Adjust based on your agent's output vocabulary and goal specificity.
- **KU-061:** Multi-goal agents (sub-goal trees) use a single goal embedding. Hierarchical progress tracking is planned for v0.2.0.
- **KU-062:** Intentional iteration (e.g., writing assistants refining across multiple drafts) may produce false positives. Set `epsilon` higher or use `criticality="LOW"` for these agents.
- **KU-063:** TF-IDF similarity may underperform when agent outputs are highly technical (code, SQL) relative to natural-language goal descriptions. Use `use_neural=True` or a custom `similarity_fn` in these cases.

---

## License

MIT — see [LICENSE](LICENSE).

---

*Pattern source: PAT-075 — John 5:5-9 (The 38-Year Stuck State)*
*"When Jesus saw him lying there and learned that he had been in this condition for a long time, he asked him, 'Do you want to get well?'"*
*The man's stuck state was structural — not laziness, not error, not idleness — but a race-condition mechanism he could never win alone.*
