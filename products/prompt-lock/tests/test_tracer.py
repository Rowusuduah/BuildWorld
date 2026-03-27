import pytest
from prompt_lock.tracer import TraceLedger


@pytest.fixture
def ledger(tmp_path):
    return TraceLedger(tmp_path / "test-traces.db")


def test_log_and_retrieve(ledger):
    ledger.log_eval(
        prompt_path="prompts/foo.txt",
        prompt_content="You are a helpful assistant.",
        eval_type="exact_match",
        score=1.0,
        passed=True,
        threshold=1.0,
        details="exact",
    )
    runs = ledger.get_recent_runs(10)
    assert len(runs) == 1
    assert runs[0]["score"] == 1.0
    assert runs[0]["passed"] is True
    assert runs[0]["eval_type"] == "exact_match"


def test_multiple_runs_ordering(ledger):
    for score in [0.5, 0.7, 0.9]:
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=score,
            passed=score >= 0.7,
            threshold=0.7,
        )
    runs = ledger.get_recent_runs(10)
    assert len(runs) == 3
    # most recent first
    assert runs[0]["score"] == 0.9


def test_get_baseline_score(ledger):
    for score in [0.80, 0.90, 0.85]:
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=score,
            passed=True,
            threshold=0.7,
        )
    baseline = ledger.get_baseline_score("p.txt", "llm_judge")
    assert baseline is not None
    assert abs(baseline - 0.85) < 0.01


def test_baseline_ignores_failed_runs(ledger):
    ledger.log_eval(
        prompt_path="p.txt",
        prompt_content="prompt",
        eval_type="llm_judge",
        score=0.30,
        passed=False,
        threshold=0.7,
    )
    baseline = ledger.get_baseline_score("p.txt", "llm_judge")
    # Only failed runs exist — no baseline
    assert baseline is None


def test_no_baseline_returns_none(ledger):
    result = ledger.get_baseline_score("nonexistent.txt", "llm_judge")
    assert result is None


def test_log_calibration(ledger):
    ledger.log_calibration(
        model="gpt-4o-mini",
        criteria="Is it good?",
        agreement_rate=0.90,
        spearman_correlation=0.85,
        bias=0.02,
        n_examples=10,
        passed=True,
    )
    # No assertion needed beyond "it doesn't raise"


def test_diff_commits(ledger):
    for commit, score in [("abc12345", 0.80), ("def67890", 0.70)]:
        ledger.log_eval(
            prompt_path="p.txt",
            prompt_content="prompt",
            eval_type="llm_judge",
            score=score,
            passed=True,
            threshold=0.7,
        )
    # We can't easily inject commit SHAs without mocking git,
    # but diff on unknown commits returns empty
    diffs = ledger.diff_commits("abc12345", "def67890")
    assert isinstance(diffs, list)
