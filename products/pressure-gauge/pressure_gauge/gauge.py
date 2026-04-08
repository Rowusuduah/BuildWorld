"""
pressure-gauge: PressureGauge class
------------------------------------
High-level API for running context pressure sweeps.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .engine import run_sweep
from .models import CriticalityLevel, PressureConfig, PressureReport


class PressureGauge:
    """
    Context pressure testing for AI agents.

    Detects behavioral drift caused by context window fill level ("context anxiety"):
    agents that summarize prematurely, rush to completion, or change behavior
    as their context window approaches capacity.

    Quick start::

        from pressure_gauge import PressureGauge, PressureConfig

        gauge = PressureGauge(PressureConfig(model_context_limit=8192))
        report = gauge.sweep(agent_fn=my_agent, base_context="Solve the problem.")
        print(report.summary())
        assert report.gate_passed, f"Context pressure: {report.recommendation}"

    """

    def __init__(
        self,
        config: Optional[PressureConfig] = None,
        *,
        model_context_limit: int = 8192,
        fill_levels: Optional[List[float]] = None,
        stability_threshold: float = 0.85,
        criticality: CriticalityLevel = CriticalityLevel.HIGH,
        padding_strategy: str = "lorem_ipsum",
        use_neural: bool = False,
    ) -> None:
        if config is not None:
            self.config = config
        else:
            kwargs: dict = {
                "model_context_limit": model_context_limit,
                "stability_threshold": stability_threshold,
                "criticality": criticality,
                "padding_strategy": padding_strategy,
            }
            if fill_levels is not None:
                kwargs["fill_levels"] = fill_levels
            self.config = PressureConfig(**kwargs)

        self._use_neural = use_neural

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def sweep(
        self,
        agent_fn: Callable[[str], str],
        base_context: str = "",
    ) -> PressureReport:
        """
        Run a full context pressure sweep across all configured fill levels.

        Parameters
        ----------
        agent_fn
            Callable that receives a (padded) context string and returns the
            agent's response string.
        base_context
            The task/query appended after the padding at every fill level.
            This ensures the agent sees the same task regardless of fill level.

        Returns
        -------
        PressureReport
            Contains ContextPressureScore, ContextDriftCurve,
            pressure_onset_token, verdict, gate_passed, and recommendation.
        """
        return run_sweep(
            config=self.config,
            agent_fn=agent_fn,
            base_context=base_context,
            use_neural=self._use_neural,
        )

    def quick(
        self,
        agent_fn: Callable[[str], str],
        base_context: str = "",
        fill_levels: Optional[List[float]] = None,
    ) -> PressureReport:
        """
        Fast sweep using only 3 fill levels (10%, 50%, 90%).

        Useful for CI smoke checks where full sweep is too slow.
        """
        levels = fill_levels if fill_levels is not None else [0.1, 0.5, 0.9]
        quick_config = PressureConfig(
            model_context_limit=self.config.model_context_limit,
            fill_levels=levels,
            stability_threshold=self.config.stability_threshold,
            criticality=self.config.criticality,
            padding_strategy=self.config.padding_strategy,
            padding_text=self.config.padding_text,
            chars_per_token=self.config.chars_per_token,
        )
        return run_sweep(
            config=quick_config,
            agent_fn=agent_fn,
            base_context=base_context,
            use_neural=self._use_neural,
        )

    def estimate_onset(
        self,
        agent_fn: Callable[[str], str],
        base_context: str = "",
        granularity: int = 10,
    ) -> Optional[int]:
        """
        Estimate pressure_onset_token at higher resolution.

        Parameters
        ----------
        granularity
            Number of evenly-spaced fill levels (default 10 → 10%, 20%, …, 100%).

        Returns
        -------
        int or None
            Approximate token count where drift first exceeds threshold,
            or None if no onset detected.
        """
        levels = [i / granularity for i in range(1, granularity + 1)]
        fine_config = PressureConfig(
            model_context_limit=self.config.model_context_limit,
            fill_levels=levels,
            stability_threshold=self.config.stability_threshold,
            criticality=self.config.criticality,
            padding_strategy=self.config.padding_strategy,
            padding_text=self.config.padding_text,
            chars_per_token=self.config.chars_per_token,
        )
        report = run_sweep(
            config=fine_config,
            agent_fn=agent_fn,
            base_context=base_context,
            use_neural=self._use_neural,
        )
        return report.pressure_onset_token
