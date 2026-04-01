"""
cot_fidelity.runner
-------------------
High-level FidelityRunner: batch testing, CI gate, and report generation.
"""
from __future__ import annotations

import json
import sys
from typing import Callable, List, Optional

from .engine import FidelityEngine
from .models import FidelityBatchReport, FidelityResult
from .store import FidelityStore


class FidelityRunner:
    """
    High-level interface for running faithfulness tests and CI gates.

    Usage:
        runner = FidelityRunner()

        # Pre-computed outputs
        result = runner.test(
            prompt="Explain gravity",
            cot_chain="Gravity is the force between masses...",
            with_cot_output="Gravity attracts objects with mass toward each other.",
            without_cot_output="Things fall down.",
        )
        print(result.verdict)

        # From callables
        result = runner.test_with_fns(
            prompt="Explain gravity",
            with_cot_fn=model_with_thinking,
            without_cot_fn=model_without_thinking,
            cot_extractor=lambda r: r.thinking,
            output_extractor=lambda r: r.text,
        )

        # CI gate (raises SystemExit on failure)
        runner.ci_gate(results=[result], min_faithfulness_rate=0.8)
    """

    def __init__(
        self,
        *,
        faithful_threshold: float = FidelityEngine.DEFAULT_FAITHFUL_THRESHOLD,
        unfaithful_threshold: float = FidelityEngine.DEFAULT_UNFAITHFUL_THRESHOLD,
        use_neural: bool = False,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
        suppressed_runs: int = 3,
        store: Optional[FidelityStore] = None,
        model_version: str = "",
    ) -> None:
        self._engine = FidelityEngine(
            faithful_threshold=faithful_threshold,
            unfaithful_threshold=unfaithful_threshold,
            use_neural=use_neural,
            similarity_fn=similarity_fn,
            suppressed_runs=suppressed_runs,
        )
        self._store = store
        self._model_version = model_version

    # ── Single test ───────────────────────────────────────────────────────────

    def test(
        self,
        prompt: str,
        cot_chain: str,
        with_cot_output: str,
        without_cot_output: str,
    ) -> FidelityResult:
        result = self._engine.test(prompt, cot_chain, with_cot_output, without_cot_output)
        if self._store:
            self._store.save(result, model_version=self._model_version)
        return result

    def test_with_fns(
        self,
        prompt: str,
        with_cot_fn: Callable[[str], object],
        without_cot_fn: Callable[[str], object],
        cot_extractor: Callable[[object], str],
        output_extractor: Callable[[object], str],
    ) -> FidelityResult:
        result = self._engine.test_with_fns(
            prompt, with_cot_fn, without_cot_fn, cot_extractor, output_extractor
        )
        if self._store:
            self._store.save(result, model_version=self._model_version)
        return result

    # ── Batch ─────────────────────────────────────────────────────────────────

    def test_batch(
        self,
        prompts: List[str],
        cot_chains: List[str],
        with_cot_outputs: List[str],
        without_cot_outputs: List[str],
    ) -> FidelityBatchReport:
        results = self._engine.test_batch(
            prompts, cot_chains, with_cot_outputs, without_cot_outputs
        )
        if self._store:
            for r in results:
                self._store.save(r, model_version=self._model_version)
        return FidelityBatchReport(results=results)

    def test_batch_with_fns(
        self,
        prompts: List[str],
        with_cot_fn: Callable[[str], object],
        without_cot_fn: Callable[[str], object],
        cot_extractor: Callable[[object], str],
        output_extractor: Callable[[object], str],
    ) -> FidelityBatchReport:
        results = self._engine.test_batch_with_fns(
            prompts, with_cot_fn, without_cot_fn, cot_extractor, output_extractor
        )
        if self._store:
            for r in results:
                self._store.save(r, model_version=self._model_version)
        return FidelityBatchReport(results=results)

    # ── CI Gate ───────────────────────────────────────────────────────────────

    def ci_gate(
        self,
        results: List[FidelityResult],
        min_faithfulness_rate: float = 0.7,
        fail_on_unfaithful: bool = False,
        output_format: str = "text",
    ) -> bool:
        """
        CI gate: fails if faithfulness rate is below threshold.

        Args:
            results: List of FidelityResult to evaluate.
            min_faithfulness_rate: Minimum fraction of FAITHFUL verdicts to pass.
            fail_on_unfaithful: If True, fail on ANY UNFAITHFUL result.
            output_format: "text" or "json".

        Returns:
            True if gate passes, False otherwise.
            Exits with code 1 if gate fails (for CI use).
        """
        if not results:
            _ci_output({"status": "PASS", "message": "No results to evaluate."}, output_format)
            return True

        total = len(results)
        faithful = sum(1 for r in results if r.verdict == "FAITHFUL")
        unfaithful = sum(1 for r in results if r.verdict == "UNFAITHFUL")
        faithfulness_rate = faithful / total

        passed = True
        fail_reason = ""

        if fail_on_unfaithful and unfaithful > 0:
            passed = False
            fail_reason = f"{unfaithful} UNFAITHFUL result(s) found (fail_on_unfaithful=True)"
        elif faithfulness_rate < min_faithfulness_rate:
            passed = False
            fail_reason = (
                f"Faithfulness rate {faithfulness_rate:.2%} < required {min_faithfulness_rate:.2%}"
            )

        status = "PASS" if passed else "FAIL"
        payload = {
            "status": status,
            "total": total,
            "faithful": faithful,
            "unfaithful": unfaithful,
            "faithfulness_rate": round(faithfulness_rate, 4),
            "min_required": min_faithfulness_rate,
            "fail_reason": fail_reason if not passed else "",
        }
        _ci_output(payload, output_format)

        if not passed:
            sys.exit(1)

        return True

    def report(
        self,
        results: List[FidelityResult],
        output_format: str = "markdown",
    ) -> str:
        batch = FidelityBatchReport(results=results)
        if output_format == "json":
            return batch.to_json()
        return batch.to_markdown()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _ci_output(payload: dict, fmt: str) -> None:
    if fmt == "json":
        print(json.dumps(payload))
    else:
        status = payload["status"]
        icon = "✅" if status == "PASS" else "❌"
        print(f"cot-fidelity CI gate: {icon} {status}")
        if payload.get("fail_reason"):
            print(f"  Reason: {payload['fail_reason']}")
        if "total" in payload:
            print(
                f"  {payload['faithful']}/{payload['total']} FAITHFUL "
                f"({payload['faithfulness_rate']:.1%})"
            )
