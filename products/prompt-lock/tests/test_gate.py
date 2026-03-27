import pytest
from prompt_lock.gate import evaluate_gate
from prompt_lock.config import GateConfig
from prompt_lock.runner import EvalResult
from prompt_lock.tracer import TraceLedger


def _result(score: float, eval_type: str = "llm_judge") -> EvalResult:
    return EvalResult(
        eval_type=eval_type,
        score=score,
        passed=score >= 0.7,
        threshold=0.7,
        details="test",
        input_text="input",
        output_text="output",
    )


@pytest.fixture
def ledger(tmp_path):
    return TraceLedger(tmp_path / "test.db")


class TestHardMode:
    def test_pass_above_threshold(self, ledger):
        cfg = GateConfig(mode="hard", hard_threshold=0.7)
        d = evaluate_gate(_result(0.85), cfg, ledger, "p.txt")
        assert d.should_fail is False
        assert d.mode == "hard"

    def test_fail_below_threshold(self, ledger):
        cfg = GateConfig(mode="hard", hard_threshold=0.7)
        d = evaluate_gate(_result(0.60), cfg, ledger, "p.txt")
        assert d.should_fail is True

    def test_exactly_at_threshold_passes(self, ledger):
        cfg = GateConfig(mode="hard", hard_threshold=0.7)
        d = evaluate_gate(_result(0.70), cfg, ledger, "p.txt")
        assert d.should_fail is False


class TestRegressionMode:
    def test_no_baseline_passes(self, ledger):
        cfg = GateConfig(mode="regression", regression_threshold=0.05)
        d = evaluate_gate(_result(0.85), cfg, ledger, "p.txt")
        assert d.should_fail is False
        assert d.baseline is None

    def test_regression_detected(self, ledger):
        # Establish baseline of 0.90
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=0.90,
            passed=True,
            threshold=0.7,
        )
        cfg = GateConfig(mode="regression", regression_threshold=0.05)
        # Score drops 0.10 — exceeds threshold of 0.05
        d = evaluate_gate(_result(0.80), cfg, ledger, "p.txt")
        assert d.should_fail is True
        assert d.baseline == pytest.approx(0.90, abs=0.01)

    def test_small_drop_ok(self, ledger):
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=0.90,
            passed=True,
            threshold=0.7,
        )
        cfg = GateConfig(mode="regression", regression_threshold=0.05)
        # Score drops 0.03 — within threshold
        d = evaluate_gate(_result(0.87), cfg, ledger, "p.txt")
        assert d.should_fail is False

    def test_improvement_never_fails(self, ledger):
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=0.70,
            passed=True,
            threshold=0.7,
        )
        cfg = GateConfig(mode="regression", regression_threshold=0.05)
        # Score improved — never fail
        d = evaluate_gate(_result(0.95), cfg, ledger, "p.txt")
        assert d.should_fail is False


class TestSoftMode:
    def test_never_fails(self, ledger):
        cfg = GateConfig(mode="soft")
        d = evaluate_gate(_result(0.0), cfg, ledger, "p.txt")
        assert d.should_fail is False
        assert d.mode == "soft"
