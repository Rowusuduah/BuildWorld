"""
Tests for livelock_probe.models
"""
import pytest
from datetime import datetime, timezone

from livelock_probe.models import (
    LIVELOCK_THRESHOLDS,
    CriticalityLevel,
    LivelockReport,
    ProgressConfig,
    StepRecord,
    get_threshold,
    make_recommendation,
    score_to_verdict,
)


# ── LIVELOCK_THRESHOLDS ───────────────────────────────────────────────────────

class TestLivelockThresholds:
    def test_all_tiers_present(self):
        assert set(LIVELOCK_THRESHOLDS.keys()) == {"CRITICAL", "HIGH", "MEDIUM", "LOW"}

    def test_thresholds_ordered(self):
        assert (
            LIVELOCK_THRESHOLDS["CRITICAL"]
            < LIVELOCK_THRESHOLDS["HIGH"]
            < LIVELOCK_THRESHOLDS["MEDIUM"]
            < LIVELOCK_THRESHOLDS["LOW"]
        )

    def test_critical_threshold(self):
        assert LIVELOCK_THRESHOLDS["CRITICAL"] == 0.05

    def test_high_threshold(self):
        assert LIVELOCK_THRESHOLDS["HIGH"] == 0.15

    def test_medium_threshold(self):
        assert LIVELOCK_THRESHOLDS["MEDIUM"] == 0.30

    def test_low_threshold(self):
        assert LIVELOCK_THRESHOLDS["LOW"] == 0.50


# ── get_threshold ─────────────────────────────────────────────────────────────

class TestGetThreshold:
    def test_critical(self):
        assert get_threshold("CRITICAL") == 0.05

    def test_high(self):
        assert get_threshold("HIGH") == 0.15

    def test_medium(self):
        assert get_threshold("MEDIUM") == 0.30

    def test_low(self):
        assert get_threshold("LOW") == 0.50


# ── score_to_verdict ──────────────────────────────────────────────────────────

class TestScoreToVerdict:
    def test_below_threshold_is_free(self):
        # 0.10 <= 0.15 (HIGH) → LIVELOCK_FREE
        assert score_to_verdict(0.10, "HIGH") == "LIVELOCK_FREE"

    def test_at_threshold_is_free(self):
        # 0.15 == 0.15 (HIGH) → LIVELOCK_FREE
        assert score_to_verdict(0.15, "HIGH") == "LIVELOCK_FREE"

    def test_borderline_above_threshold(self):
        # 0.18 is above 0.15 but within band 0.05 → BORDERLINE
        assert score_to_verdict(0.18, "HIGH") == "BORDERLINE"

    def test_at_borderline_edge_is_borderline(self):
        # 0.20 = 0.15 + 0.05 → BORDERLINE (edge of band)
        assert score_to_verdict(0.20, "HIGH") == "BORDERLINE"

    def test_above_borderline_is_detected(self):
        # 0.21 > 0.15 + 0.05 → LIVELOCK_DETECTED
        assert score_to_verdict(0.21, "HIGH") == "LIVELOCK_DETECTED"

    def test_critical_strict(self):
        # 0.06 > 0.05 → BORDERLINE (within 0.05 band)
        assert score_to_verdict(0.06, "CRITICAL") == "BORDERLINE"

    def test_critical_far_above(self):
        # 0.15 >> 0.05 threshold → LIVELOCK_DETECTED
        assert score_to_verdict(0.15, "CRITICAL") == "LIVELOCK_DETECTED"

    def test_zero_score_is_always_free(self):
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            assert score_to_verdict(0.0, tier) == "LIVELOCK_FREE"  # type: ignore

    def test_perfect_livelock_is_detected(self):
        # score 1.0 = 100% stuck → LIVELOCK_DETECTED for all tiers
        for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            assert score_to_verdict(1.0, tier) == "LIVELOCK_DETECTED"  # type: ignore


# ── ProgressConfig ────────────────────────────────────────────────────────────

class TestProgressConfig:
    def test_valid_config(self):
        cfg = ProgressConfig(goal="fix the bug")
        assert cfg.goal == "fix the bug"
        assert cfg.k == 5
        assert cfg.epsilon == 0.05
        assert cfg.criticality == "HIGH"
        assert cfg.budget_steps == 100

    def test_empty_goal_raises(self):
        with pytest.raises(ValueError, match="goal"):
            ProgressConfig(goal="")

    def test_whitespace_goal_raises(self):
        with pytest.raises(ValueError, match="goal"):
            ProgressConfig(goal="   ")

    def test_k_zero_raises(self):
        with pytest.raises(ValueError, match="k"):
            ProgressConfig(goal="fix bug", k=0)

    def test_k_negative_raises(self):
        with pytest.raises(ValueError, match="k"):
            ProgressConfig(goal="fix bug", k=-1)

    def test_epsilon_zero_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            ProgressConfig(goal="fix bug", epsilon=0.0)

    def test_epsilon_one_raises(self):
        with pytest.raises(ValueError, match="epsilon"):
            ProgressConfig(goal="fix bug", epsilon=1.0)

    def test_budget_steps_zero_raises(self):
        with pytest.raises(ValueError, match="budget_steps"):
            ProgressConfig(goal="fix bug", budget_steps=0)

    def test_custom_k(self):
        cfg = ProgressConfig(goal="do task", k=10)
        assert cfg.k == 10

    def test_similarity_fn_stored(self):
        fn = lambda a, b: 1.0
        cfg = ProgressConfig(goal="do task", similarity_fn=fn)
        assert cfg.similarity_fn is fn


# ── make_recommendation ───────────────────────────────────────────────────────

class TestMakeRecommendation:
    def test_no_livelock_low_score(self):
        rec = make_recommendation(0.02, False, 2, 5, None)
        assert "progressing normally" in rec.lower()

    def test_no_livelock_medium_score(self):
        rec = make_recommendation(0.12, False, 3, 5, None)
        assert "mostly progressing" in rec.lower()

    def test_livelock_detected(self):
        rec = make_recommendation(0.80, True, 8, 5, 2)
        assert "LIVELOCK DETECTED" in rec
        assert "step 2" in rec
        assert "8" in rec

    def test_livelock_no_window_start(self):
        rec = make_recommendation(0.80, True, 8, 5, None)
        assert "LIVELOCK DETECTED" in rec
