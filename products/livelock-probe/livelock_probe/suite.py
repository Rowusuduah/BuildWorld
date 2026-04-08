"""
livelock_probe.suite
--------------------
Stateful LivelockSuite — records agent step outputs and computes livelock on demand.

Two usage patterns:

  1. Manual instrumentation:
       suite = LivelockSuite(ProgressConfig(goal="resolve DB error", k=5))
       for output in agent_outputs:
           suite.record_step(output)
           if not suite.gate():
               break  # agent is in livelock — stop early
       report = suite.compute()

  2. Context manager (instrument a block):
       suite = LivelockSuite(ProgressConfig(goal="resolve DB error", k=5))
       with suite.monitor() as monitor:
           for _ in range(max_steps):
               output = agent.run()
               monitor.record(output)
       report = suite.compute()
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator, List, Optional

from .engine import LivelockEngine
from .models import LivelockReport, ProgressConfig


class _StepMonitor:
    """Context object returned by LivelockSuite.monitor()."""

    def __init__(self, suite: "LivelockSuite") -> None:
        self._suite = suite

    def record(self, step_output: str, step_id: Optional[int] = None) -> None:
        """Record a step output. step_id defaults to next sequential index."""
        self._suite.record_step(step_output, step_id=step_id)


class LivelockSuite:
    """
    Stateful livelock detection for a single agent run.

    Thread-safety: LivelockSuite is NOT thread-safe. For concurrent agent runs,
    create one LivelockSuite per run.

    Args:
        config: ProgressConfig specifying goal, k, epsilon, and criticality.
        engine: Optional custom LivelockEngine. Defaults to standard engine.
    """

    def __init__(
        self,
        config: ProgressConfig,
        engine: Optional[LivelockEngine] = None,
    ) -> None:
        self._config = config
        self._engine = engine or LivelockEngine(
            use_neural=config.use_neural,
            similarity_fn=config.similarity_fn,
        )
        self._steps: List[str] = []
        self._last_report: Optional[LivelockReport] = None

    # ── Step Recording ────────────────────────────────────────────────────────

    def record_step(self, step_output: str, step_id: Optional[int] = None) -> None:
        """
        Record a single agent step output.

        Args:
            step_output: String output produced by the agent at this step.
            step_id: Optional explicit step ID. If provided, must match the
                     expected next index. Gaps are not allowed; use None for
                     automatic sequential indexing.

        Raises:
            ValueError: If step_id is provided but does not match expected index.
        """
        if step_id is not None and step_id != len(self._steps):
            raise ValueError(
                f"step_id {step_id} does not match expected index {len(self._steps)}. "
                f"Steps must be recorded in order. Use step_id=None for automatic indexing."
            )
        self._steps.append(step_output)
        self._last_report = None  # invalidate cached report

    def record_steps(self, step_outputs: List[str]) -> None:
        """
        Record multiple step outputs at once.

        Args:
            step_outputs: List of step output strings in order.
        """
        for output in step_outputs:
            self.record_step(output)

    # ── Compute ───────────────────────────────────────────────────────────────

    def compute(self) -> LivelockReport:
        """
        Compute LivelockReport from all recorded steps.

        Returns:
            LivelockReport with full livelock analysis.

        Raises:
            ValueError: If no steps have been recorded.
        """
        if not self._steps:
            raise ValueError(
                "No steps recorded. Call record_step() before compute()."
            )
        if self._last_report is not None:
            return self._last_report
        report = self._engine.compute(self._steps, self._config)
        self._last_report = report
        return report

    def gate(self) -> bool:
        """
        Return True if the current step sequence passes the livelock gate.

        Computes a fresh report and checks whether livelock_score <= threshold.
        Returns True (safe) if fewer than k consecutive stuck steps detected.

        Returns:
            True if gate passes (no livelock above threshold).
            False if livelock detected above threshold.

        Raises:
            ValueError: If no steps have been recorded.
        """
        report = self.compute()
        return report.gate_passed

    def last_report(self) -> Optional[LivelockReport]:
        """Return the most recently computed report, or None if not yet computed."""
        return self._last_report

    def step_count(self) -> int:
        """Return the number of steps recorded so far."""
        return len(self._steps)

    def is_over_budget(self) -> bool:
        """Return True if step count exceeds config.budget_steps."""
        return len(self._steps) >= self._config.budget_steps

    # ── Reset ─────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """
        Clear all recorded steps and cached report.
        Use this to reuse the suite for a new agent run with the same config.
        """
        self._steps = []
        self._last_report = None

    # ── Context Manager ───────────────────────────────────────────────────────

    @contextmanager
    def monitor(self) -> Generator[_StepMonitor, None, None]:
        """
        Context manager for step recording.

        Usage:
            with suite.monitor() as m:
                for _ in range(max_steps):
                    output = agent.run()
                    m.record(output)
            report = suite.compute()
        """
        monitor = _StepMonitor(self)
        try:
            yield monitor
        finally:
            pass  # state lives in suite; nothing to clean up here
