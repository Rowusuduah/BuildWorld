"""
cot-fidelity: Measure whether your LLM's reasoning chain actually caused its output.
====================================================================================

The first pip-installable tool that implements the counterfactual suppression test
for chain-of-thought (CoT) faithfulness — detecting when a model's stated reasoning
is unfaithful to its actual computation.

Inspired by Anthropic's 2025 paper "Reasoning Models Don't Always Say What They Think":
  "If the CoT is not faithful, we cannot depend on our ability to monitor CoT in order
   to detect misaligned behaviors, because there may be safety-relevant factors affecting
   model behavior that have not been explicitly verbalized."

Algorithm:
  1. Run model WITH reasoning chain → full_output
  2. Run model WITHOUT reasoning chain (suppressed) → suppressed_output
  3. faithfulness_score = 1 - cosine_similarity(full_output, suppressed_output)
  4. High score → CoT WAS causal → FAITHFUL
     Low score  → CoT was NOT causal → UNFAITHFUL

Quick start:
    from cot_fidelity import FidelityEngine

    engine = FidelityEngine()
    result = engine.test(
        prompt="Explain why water boils at 100°C",
        cot_chain="I need to think about thermodynamics...",
        with_cot_output="Water boils at 100°C due to vapor pressure equaling atmospheric pressure.",
        without_cot_output="Water boils at 100°C.",
    )
    print(result.verdict)          # FAITHFUL / UNFAITHFUL / INCONCLUSIVE
    print(result.faithfulness_score)  # 0.0 – 1.0

Author: BuildWorld — Cycle 020
Biblical Foundation: PAT-059 (Genesis 3:1-6, score 10.0/10 — first perfect pattern)
License: MIT
"""
from __future__ import annotations

from .engine import FidelityEngine
from .models import (
    DriftPoint,
    DriftReport,
    FidelityBatchReport,
    FidelityResult,
    FidelityVerdict,
)
from .runner import FidelityRunner
from .store import FidelityStore
from .decorators import (
    UnfaithfulCoTError,
    faithfulness_probe,
    faithfulness_probe_pair,
)

__version__ = "0.1.0"
__all__ = [
    # Engine
    "FidelityEngine",
    # Runner
    "FidelityRunner",
    # Store
    "FidelityStore",
    # Models
    "FidelityResult",
    "FidelityBatchReport",
    "DriftPoint",
    "DriftReport",
    "FidelityVerdict",
    # Decorators
    "faithfulness_probe",
    "faithfulness_probe_pair",
    "UnfaithfulCoTError",
]
