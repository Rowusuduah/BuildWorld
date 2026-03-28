# agent-patrol

> Runtime pathology detection for AI agents. Diagnoses loops, stalls, oscillation, drift, and silent abandonment.

[![PyPI](https://img.shields.io/pypi/v/agent-patrol)](https://pypi.org/project/agent-patrol/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## The Problem

Your AI agents get stuck. They loop, oscillate, stall, drift from their task, and silently abandon their mission -- and you don't know until the token bill arrives.

Current solutions are blunt instruments:
- **max_steps / timeout** -- kills productive and unproductive agents equally
- **Observability platforms** -- trace what happened but not why it went wrong
- **Framework-specific guards** -- locked to one framework, not portable

**agent-patrol** diagnoses *what kind* of failure your agent is experiencing, in real time, across any framework.

## Five Agent Pathologies

| # | Pathology | What It Looks Like | What agent-patrol Does |
|---|-----------|-------------------|----------------------|
| 1 | **Futile Cycle** | Agent repeats the same action with minor variations | Measures semantic similarity between consecutive actions. Flags when iterations stop producing new information. |
| 2 | **Oscillation** | Agent alternates between two contradictory sub-goals | Clusters actions into goal-groups. Detects when the agent is "double-minded" -- switching between conflicting objectives. |
| 3 | **Stall** | Agent is active but making no progress toward its goal | Tracks semantic distance to declared milestones over time. Distinguishes "legitimately slow" from "stuck." |
| 4 | **Drift** | Agent gradually wanders from its original task | Compares current action context to the original task embedding. Flags when intent has shifted beyond threshold. |
| 5 | **Abandonment** | Agent silently stops working on its task and does something else | Detects combined low-similarity-to-original-task + high-coherence-with-novel-objective. |

## Installation

```bash
pip install agent-patrol
```

**Zero hard dependencies.** Works with Python stdlib alone. Optional enhancements:

```bash
# For embedding-based detection (recommended for production)
pip install agent-patrol[embeddings]

# For LLM-based semantic analysis
pip install agent-patrol[llm]
```

## Quick Start

### Option 1: Decorator

```python
from agent_patrol import patrol

@patrol(on_pathology="log", sensitivity="medium")
def agent_step(state: dict) -> dict:
    # Your agent logic here
    return new_state
```

### Option 2: Monitor

```python
from agent_patrol import PatrolMonitor

monitor = PatrolMonitor(
    task_description="Research and summarize recent economic data",
    milestones=["Find GDP data", "Find inflation data", "Write summary"],
)

for step in agent_loop():
    report = monitor.observe(step.action, step.result)
    if report.pathology:
        print(f"DETECTED: {report.pathology} at step {report.step_detected}")
        print(f"Evidence: {report.evidence}")
        print(f"Action: {report.recommended_action}")
```

### Option 3: CI Gate

```bash
# Run agent-patrol on a recorded trace file
agent-patrol check trace.jsonl --fail-on FUTILE_CYCLE,STALL
```

## Output

```python
PatrolReport(
    pathology="FUTILE_CYCLE",
    confidence=0.87,
    step_detected=14,
    evidence=["Steps 12-14: repeated 'search for X' with <0.05 semantic variance"],
    recommended_action="HALT",
    total_steps_observed=14,
    token_estimate_wasted=3400,
)
```

## Framework Integration

Works with any agent framework. No lock-in.

- **LangGraph** -- wrap your node functions
- **CrewAI** -- wrap agent step callbacks
- **OpenAI Agents SDK** -- wrap tool execution
- **Anthropic Claude Agent SDK** -- wrap tool use blocks
- **Raw loops** -- wrap your iteration function

See [examples/](examples/) for integration guides.

## Configuration

```python
from agent_patrol import PatrolConfig

config = PatrolConfig(
    # Detection sensitivity (low = fewer false positives, high = catches more)
    sensitivity="medium",  # "low" | "medium" | "high"

    # Which pathologies to detect
    detect=["FUTILE_CYCLE", "OSCILLATION", "STALL", "DRIFT", "ABANDONMENT"],

    # What to do when pathology is detected
    on_pathology="log",  # "log" | "raise" | "halt" | "callback"

    # Similarity backend
    similarity_backend="tfidf",  # "tfidf" | "embeddings" | "llm"

    # Custom thresholds
    cycle_similarity_threshold=0.92,
    stall_window=5,
    drift_threshold=0.60,
    oscillation_min_switches=3,
)
```

## How It Works

agent-patrol maintains a rolling window of agent actions (text descriptions + optional state). For each new action, it runs five lightweight detectors:

1. **Futile Cycle Detector** -- Computes pairwise similarity in a sliding window. If the last N actions are all >0.92 similar, the agent is cycling.

2. **Oscillation Detector** -- Clusters actions by semantic similarity. If the agent alternates between 2+ clusters more than 3 times, it is oscillating.

3. **Stall Detector** -- If milestones are declared, computes distance to each milestone at every step. If the minimum distance is non-decreasing for 5+ steps, the agent is stalled.

4. **Drift Detector** -- Maintains the original task embedding. Computes cosine similarity between current action context and original task. If below 0.60, intent has drifted.

5. **Abandonment Detector** -- Combines drift detection with novel-cluster detection. If the agent is far from its original task AND close to a new coherent objective, it has silently abandoned its task.

## Why This Exists

Every team running AI agents in production has experienced stuck agents. The token cost of a stuck agent looping for 200 steps before a timeout kills it can be $5-50 per incident. Multiply by hundreds of daily agent runs and the cost is material.

More importantly: stuck agents produce bad output. A stalled research agent returns incomplete results. A drifted coding agent implements the wrong feature. An oscillating planning agent produces an incoherent plan. These failures are silent -- the agent returns output that looks complete but isn't.

agent-patrol makes these failures visible and actionable.

## License

MIT
