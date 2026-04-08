"""
Tests for livelock_probe.engine
"""
import math
import pytest

from livelock_probe.engine import (
    LivelockEngine,
    _compute_progress_deltas,
    _compute_progress_vector,
    _find_max_consecutive_stuck,
    _find_stuck_window,
    _tfidf_cosine,
    _tokenize,
)
from livelock_probe.models import ProgressConfig


# ── _tokenize ─────────────────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        assert _tokenize("Hello World") == ["hello", "world"]

    def test_numbers(self):
        assert "42" in _tokenize("step 42 complete")

    def test_special_chars_excluded(self):
        tokens = _tokenize("fix: bug-report (critical)")
        assert "fix" in tokens
        assert "bug" in tokens
        assert "critical" in tokens
        assert ":" not in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_repeated_words(self):
        tokens = _tokenize("error error error")
        assert tokens.count("error") == 3


# ── _tfidf_cosine ─────────────────────────────────────────────────────────────

class TestTfidfCosine:
    def test_identical_texts(self):
        score = _tfidf_cosine("resolve database error", "resolve database error")
        assert score > 0.99

    def test_empty_texts_return_zero(self):
        assert _tfidf_cosine("", "hello world") == 0.0
        assert _tfidf_cosine("hello world", "") == 0.0
        assert _tfidf_cosine("", "") == 0.0

    def test_similar_texts_score_higher_than_unrelated(self):
        sim_high = _tfidf_cosine(
            "fix the database connection error",
            "resolved the database connection issue",
        )
        sim_low = _tfidf_cosine(
            "fix the database connection error",
            "the weather is nice today",
        )
        assert sim_high > sim_low

    def test_result_in_range(self):
        score = _tfidf_cosine("some text here", "another text here")
        assert 0.0 <= score <= 1.0

    def test_completely_different_texts(self):
        score = _tfidf_cosine("alpha beta gamma", "delta epsilon zeta")
        assert score == 0.0  # no overlap


# ── _compute_progress_vector ──────────────────────────────────────────────────

class TestComputeProgressVector:
    def _fixed_sim(self, values):
        """Build a deterministic similarity function from a list of expected values."""
        calls = iter(values)
        return lambda a, b: next(calls)

    def test_progress_vector_length(self):
        sim_fn = lambda a, b: 0.5
        steps = ["step 1", "step 2", "step 3"]
        vec = _compute_progress_vector(steps, "goal", sim_fn)
        assert len(vec) == 3

    def test_progress_vector_values(self):
        values = [0.1, 0.5, 0.9]
        sim_fn = self._fixed_sim(values)
        vec = _compute_progress_vector(["a", "b", "c"], "goal", sim_fn)
        assert vec == pytest.approx([0.1, 0.5, 0.9])

    def test_empty_steps(self):
        sim_fn = lambda a, b: 0.5
        vec = _compute_progress_vector([], "goal", sim_fn)
        assert vec == []


# ── _compute_progress_deltas ──────────────────────────────────────────────────

class TestComputeProgressDeltas:
    def test_first_delta_equals_first_value(self):
        deltas = _compute_progress_deltas([0.3, 0.6, 0.9])
        assert deltas[0] == pytest.approx(0.3)

    def test_subsequent_deltas(self):
        deltas = _compute_progress_deltas([0.3, 0.6, 0.9])
        assert deltas[1] == pytest.approx(0.3)
        assert deltas[2] == pytest.approx(0.3)

    def test_negative_delta_when_regressing(self):
        deltas = _compute_progress_deltas([0.8, 0.5, 0.3])
        assert deltas[1] < 0
        assert deltas[2] < 0

    def test_zero_delta_when_stuck(self):
        deltas = _compute_progress_deltas([0.5, 0.5, 0.5])
        # First delta = 0.5; subsequent deltas = 0.0
        assert deltas[1] == pytest.approx(0.0)
        assert deltas[2] == pytest.approx(0.0)

    def test_empty_returns_empty(self):
        assert _compute_progress_deltas([]) == []

    def test_single_value(self):
        deltas = _compute_progress_deltas([0.7])
        assert deltas == pytest.approx([0.7])


# ── _find_max_consecutive_stuck ────────────────────────────────────────────────

class TestFindMaxConsecutiveStuck:
    def test_no_stuck(self):
        assert _find_max_consecutive_stuck([False, False, False]) == 0

    def test_all_stuck(self):
        assert _find_max_consecutive_stuck([True, True, True]) == 3

    def test_single_stuck(self):
        assert _find_max_consecutive_stuck([False, True, False]) == 1

    def test_two_windows(self):
        # [T, T, F, T, T, T] → longest run = 3
        assert _find_max_consecutive_stuck([True, True, False, True, True, True]) == 3

    def test_empty(self):
        assert _find_max_consecutive_stuck([]) == 0


# ── _find_stuck_window ────────────────────────────────────────────────────────

class TestFindStuckWindow:
    def test_no_stuck(self):
        start, end = _find_stuck_window([False, False, False])
        assert start is None
        assert end is None

    def test_single_window(self):
        start, end = _find_stuck_window([False, True, True, True, False])
        assert start == 1
        assert end == 3

    def test_two_windows_returns_longest(self):
        # [T, F, T, T, T] → first window is step 0 (len 1), second is steps 2-4 (len 3)
        start, end = _find_stuck_window([True, False, True, True, True])
        assert start == 2
        assert end == 4

    def test_all_stuck(self):
        start, end = _find_stuck_window([True, True, True])
        assert start == 0
        assert end == 2


# ── LivelockEngine ────────────────────────────────────────────────────────────

class TestLivelockEngine:
    def _always_sim(self, val: float):
        return lambda a, b: val

    def test_progressing_agent_no_livelock(self):
        """Agent that increases similarity to goal → no livelock."""
        config = ProgressConfig(goal="resolve error", k=3, similarity_fn=None)
        # Simulate increasing progress
        step_vals = [0.1, 0.3, 0.5, 0.7, 0.9]
        engine = LivelockEngine(similarity_fn=self._always_sim(0.5))
        # Use custom injected fn that returns increasing values via sequence
        seq = iter(step_vals)
        config2 = ProgressConfig(goal="resolve error", k=3, similarity_fn=lambda a, b: next(seq))
        report = engine.compute(["s1", "s2", "s3", "s4", "s5"], config2)
        assert report.total_steps == 5
        assert report.livelock_detected is False

    def test_fully_stuck_agent(self):
        """Agent that returns identical output every step → livelock detected."""
        # All similarities identical → deltas near 0 after step 0
        sim_seq = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
        seq = iter(sim_seq)
        config = ProgressConfig(goal="resolve error", k=3, similarity_fn=lambda a, b: next(seq))
        engine = LivelockEngine()
        report = engine.compute(["retry"] * 6, config)
        assert report.livelock_detected is True
        assert report.max_consecutive_stuck >= 3

    def test_livelock_score_all_stuck(self):
        """If all steps (except first) are stuck, livelock_score should be high."""
        # delta[0] = 0.5 (not stuck if >= epsilon), delta[1..4] = 0.0 (stuck)
        sims = [0.5, 0.5, 0.5, 0.5, 0.5]
        seq = iter(sims)
        config = ProgressConfig(goal="do task", k=3, epsilon=0.05, similarity_fn=lambda a, b: next(seq))
        engine = LivelockEngine()
        report = engine.compute(["x"] * 5, config)
        # Steps 1-4 have delta=0.0 which is < epsilon → stuck
        assert report.livelock_score > 0.5

    def test_gate_passes_for_progressing_agent(self):
        """Gate passes when livelock_score is within threshold."""
        sims = [0.1, 0.3, 0.5, 0.7, 0.9]
        seq = iter(sims)
        config = ProgressConfig(goal="do task", k=3, similarity_fn=lambda a, b: next(seq))
        engine = LivelockEngine()
        report = engine.compute(["x"] * 5, config)
        assert report.gate_passed is True

    def test_empty_steps_raises(self):
        config = ProgressConfig(goal="do task")
        engine = LivelockEngine()
        with pytest.raises(ValueError, match="at least 1 step"):
            engine.compute([], config)

    def test_report_has_all_fields(self):
        config = ProgressConfig(goal="resolve issue", k=3)
        engine = LivelockEngine(similarity_fn=lambda a, b: 0.5)
        report = engine.compute(["step1", "step2", "step3"], config)
        assert report.report_id
        assert report.goal == "resolve issue"
        assert report.total_steps == 3
        assert len(report.steps) == 3
        assert len(report.progress_vector) == 3
        assert len(report.progress_deltas) == 3
        assert report.recommendation
        assert report.tested_at is not None

    def test_stuck_window_identified_correctly(self):
        """The stuck window should point to the longest consecutive stuck block."""
        # Steps 0-1 progress, steps 2-6 stuck, step 7 progresses
        sims = [0.1, 0.3, 0.35, 0.36, 0.365, 0.37, 0.371, 0.8]
        seq = iter(sims)
        config = ProgressConfig(
            goal="resolve issue", k=3, epsilon=0.05,
            similarity_fn=lambda a, b: next(seq),
        )
        engine = LivelockEngine()
        report = engine.compute(["x"] * 8, config)
        # Steps 2-6 all have very small deltas → stuck
        assert report.stuck_window_start is not None

    def test_summary_string(self):
        config = ProgressConfig(goal="do task", k=3)
        engine = LivelockEngine(similarity_fn=lambda a, b: 0.5)
        report = engine.compute(["s1", "s2", "s3"], config)
        summary = report.summary()
        assert "livelock_score" in summary
        assert "criticality" in summary

    def test_to_dict_serialisable(self):
        import json
        config = ProgressConfig(goal="do task", k=3)
        engine = LivelockEngine(similarity_fn=lambda a, b: 0.5)
        report = engine.compute(["s1", "s2", "s3"], config)
        d = report.to_dict()
        # Should be JSON serialisable
        json.dumps(d)
        assert d["goal"] == "do task"
        assert "livelock_score" in d

    def test_injectable_similarity_fn_in_engine(self):
        """Engine-level similarity_fn is used when config has none."""
        always_high = lambda a, b: 0.9
        engine = LivelockEngine(similarity_fn=always_high)
        config = ProgressConfig(goal="goal", k=2)
        report = engine.compute(["x", "y", "z"], config)
        assert all(v == pytest.approx(0.9) for v in report.progress_vector)

    def test_config_similarity_fn_overrides_engine(self):
        """Config's similarity_fn takes priority over engine's."""
        engine_fn = lambda a, b: 0.1
        config_fn = lambda a, b: 0.8
        engine = LivelockEngine(similarity_fn=engine_fn)
        config = ProgressConfig(goal="goal", k=2, similarity_fn=config_fn)
        report = engine.compute(["x", "y"], config)
        assert all(v == pytest.approx(0.8) for v in report.progress_vector)
