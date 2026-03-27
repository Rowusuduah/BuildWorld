"""CI gate logic: decide whether an eval result should fail the build."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import GateConfig
from .runner import EvalResult
from .tracer import TraceLedger


@dataclass
class GateDecision:
    should_fail: bool
    reason: str
    score: float
    baseline: Optional[float]
    mode: str


def evaluate_gate(
    result: EvalResult,
    config: GateConfig,
    tracer: TraceLedger,
    prompt_path: str,
) -> GateDecision:
    """Determine whether this eval result should fail the CI gate.

    Modes:
        hard       — fail if score < hard_threshold (absolute floor)
        regression — fail if score dropped > regression_threshold from recent baseline
        soft       — never fail, warn only (useful for new prompts)
    """
    if config.mode == "hard":
        should_fail = result.score < config.hard_threshold
        return GateDecision(
            should_fail=should_fail,
            reason=(
                f"score {result.score:.3f} "
                f"{'<' if should_fail else '>='} hard threshold {config.hard_threshold}"
            ),
            score=result.score,
            baseline=None,
            mode="hard",
        )

    elif config.mode == "regression":
        baseline = tracer.get_baseline_score(prompt_path, result.eval_type)
        if baseline is None:
            # No baseline yet — this is the first run; record and pass
            return GateDecision(
                should_fail=False,
                reason=f"no baseline yet — recording score {result.score:.3f} as initial baseline",
                score=result.score,
                baseline=None,
                mode="regression",
            )
        drop = baseline - result.score
        should_fail = drop > config.regression_threshold
        return GateDecision(
            should_fail=should_fail,
            reason=(
                f"score {result.score:.3f}, baseline {baseline:.3f}, "
                f"drop {drop:.3f} "
                f"({'> threshold' if should_fail else '<= threshold'} {config.regression_threshold})"
            ),
            score=result.score,
            baseline=baseline,
            mode="regression",
        )

    else:  # soft
        return GateDecision(
            should_fail=False,
            reason=f"soft mode — score {result.score:.3f} (warning only)",
            score=result.score,
            baseline=None,
            mode="soft",
        )
