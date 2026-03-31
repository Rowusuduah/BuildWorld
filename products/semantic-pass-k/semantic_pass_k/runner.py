"""
semantic_pass_k.runner
----------------------
ConsistencyRunner: orchestrates running an agent k times and evaluating consistency.

Separates concerns:
  - ConsistencyEngine: pure algorithm (embed + score)
  - ConsistencyRunner: orchestration (call agent, collect outputs, delegate to engine)
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .engine import ConsistencyEngine
from .models import (
    CriticalityLevel,
    ConsistencyReport,
    ConsistencyResult,
)


class ConsistencyRunner:
    """
    Run an agent function k times and measure semantic output consistency.

    Example:
        def my_agent(prompt: str) -> str:
            return call_llm(prompt)

        runner = ConsistencyRunner(agent_fn=my_agent, k=5)
        result = runner.run(
            prompt="Summarise the Ghana Investment Policy 2024.",
            criticality="HIGH",
        )
        print(result.verdict)   # CONSISTENT / BORDERLINE / INCONSISTENT
        print(result.consistency_score)

    Tip: For testing without an LLM, pass a deterministic or stubbed agent_fn.
    """

    def __init__(
        self,
        agent_fn: Callable[[str], str],
        *,
        k: int = 5,
        criticality: CriticalityLevel = "HIGH",
        engine: Optional[ConsistencyEngine] = None,
        agent_label: str = "default",
    ) -> None:
        """
        Args:
            agent_fn: Callable that takes a prompt string and returns a string output.
            k: Number of times to run the agent per prompt (default: 5).
               Must be >= 2.
            criticality: Default criticality tier for threshold selection.
                         Can be overridden per run() call.
            engine: Optional pre-configured ConsistencyEngine. If None,
                    a default engine is created (TF-IDF cosine, zero deps).
            agent_label: Human-readable label stored in results (e.g. "gpt-4o").
        """
        if k < 2:
            raise ValueError(f"k must be >= 2. Got {k}.")
        self.agent_fn = agent_fn
        self.k = k
        self.default_criticality = criticality
        self.engine = engine or ConsistencyEngine(agent_label=agent_label)
        self.agent_label = agent_label

    def run(
        self,
        prompt: str,
        *,
        k: Optional[int] = None,
        criticality: Optional[CriticalityLevel] = None,
    ) -> ConsistencyResult:
        """
        Run the agent k times on the prompt and evaluate consistency.

        Args:
            prompt: The prompt to evaluate.
            k: Override the runner's default k for this run.
            criticality: Override the runner's default criticality for this run.

        Returns:
            ConsistencyResult
        """
        effective_k = k if k is not None else self.k
        effective_criticality = criticality or self.default_criticality

        if effective_k < 2:
            raise ValueError(f"k must be >= 2. Got {effective_k}.")

        outputs = [self.agent_fn(prompt) for _ in range(effective_k)]
        return self.engine.evaluate(prompt, outputs, effective_criticality)

    def run_batch(
        self,
        prompts: List[str],
        *,
        k: Optional[int] = None,
        criticality: Optional[CriticalityLevel] = None,
        label: str = "consistency_report",
    ) -> ConsistencyReport:
        """
        Run the agent k times on each prompt and return an aggregate report.

        Args:
            prompts: List of prompts to evaluate.
            k: Override k for all runs in this batch.
            criticality: Override criticality for all runs in this batch.
            label: Human-readable label for the report.

        Returns:
            ConsistencyReport with per-prompt results and aggregate statistics.
        """
        if not prompts:
            raise ValueError("prompts list cannot be empty.")

        results = [
            self.run(p, k=k, criticality=criticality)
            for p in prompts
        ]
        return ConsistencyReport.from_results(results, label=label)
