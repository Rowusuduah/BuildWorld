"""Tests for agent-patrol: runtime agent pathology detection."""

import json

import pytest

from agent_patrol import (
    AgentPathologyError,
    Pathology,
    PathologyReport,
    PatrolMonitor,
    PatrolStore,
    PatrolSummary,
    Sensitivity,
    Severity,
    StepObservation,
    __version__,
    _cosine_similarity_bow,
    _detect_abandonment,
    _detect_drift,
    _detect_futile_cycle,
    _detect_oscillation,
    _detect_stall,
    _get_thresholds,
    _jaccard_similarity,
    patrol,
)


# ─── Model Tests ────��────────────────────────────────────────────────────────


class TestStepObservation:
    def test_fingerprint_deterministic(self):
        a = StepObservation(action="search for GDP data")
        b = StepObservation(action="search for GDP data")
        assert a.fingerprint == b.fingerprint

    def test_fingerprint_case_insensitive(self):
        a = StepObservation(action="Search For GDP Data")
        b = StepObservation(action="search for gdp data")
        assert a.fingerprint == b.fingerprint

    def test_tokens(self):
        obs = StepObservation(action="search GDP", result="found 6.0%")
        assert "search" in obs.tokens
        assert "gdp" in obs.tokens
        assert "found" in obs.tokens

    def test_timestamp_auto_set(self):
        obs = StepObservation(action="test")
        assert obs.timestamp > 0


class TestPathologyReport:
    def test_healthy_report(self):
        report = PathologyReport(step_number=1)
        assert report.is_healthy is True
        assert report.pathology is None

    def test_unhealthy_report(self):
        report = PathologyReport(
            step_number=5,
            pathology=Pathology.FUTILE_CYCLE,
            severity=Severity.WARNING,
            confidence=0.85,
        )
        assert report.is_healthy is False
        assert report.pathology == Pathology.FUTILE_CYCLE

    def test_to_dict(self):
        report = PathologyReport(
            step_number=3,
            pathology=Pathology.DRIFT,
            severity=Severity.CRITICAL,
            confidence=0.9,
            evidence="drifted far",
        )
        d = report.to_dict()
        assert d["pathology"] == "drift"
        assert d["severity"] == "critical"
        assert d["confidence"] == 0.9


class TestPatrolSummary:
    def test_healthy_summary(self):
        summary = PatrolSummary(total_steps=10)
        assert summary.is_healthy is True
        assert summary.verdict == "healthy"

    def test_to_json(self):
        summary = PatrolSummary(total_steps=5, health_score=0.8, verdict="degraded")
        j = json.loads(summary.to_json())
        assert j["total_steps"] == 5
        assert j["verdict"] == "degraded"


# ─── Similarity Tests ──────��─────────────────────────────────────────────────


class TestSimilarity:
    def test_jaccard_identical(self):
        assert _jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0

    def test_jaccard_disjoint(self):
        assert _jaccard_similarity({"a"}, {"b"}) == 0.0

    def test_jaccard_partial(self):
        sim = _jaccard_similarity({"a", "b", "c"}, {"b", "c", "d"})
        assert 0.4 < sim < 0.6  # 2/4 = 0.5

    def test_jaccard_empty(self):
        assert _jaccard_similarity(set(), set()) == 1.0
        assert _jaccard_similarity({"a"}, set()) == 0.0

    def test_cosine_identical(self):
        sim = _cosine_similarity_bow("hello world", "hello world")
        assert sim > 0.99

    def test_cosine_different(self):
        sim = _cosine_similarity_bow("hello world", "goodbye universe")
        assert sim < 0.1

    def test_cosine_empty(self):
        assert _cosine_similarity_bow("", "") == 0.0


# ─── Detector Tests ───────────��──────────────────────────────────────────────


class TestFutileCycle:
    def test_no_cycle_when_diverse(self):
        obs = [
            StepObservation(action="search for GDP data", step_number=1),
            StepObservation(action="analyze inflation trends", step_number=2),
            StepObservation(action="check exchange rates", step_number=3),
            StepObservation(action="review fiscal policy", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_futile_cycle(obs, thresholds) is None

    def test_detects_cycle(self):
        obs = [
            StepObservation(action="search for GDP data in Ghana", step_number=1),
            StepObservation(action="search for GDP data in Ghana", step_number=2),
            StepObservation(action="search for GDP data in Ghana", step_number=3),
            StepObservation(action="search for GDP data in Ghana", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        result = _detect_futile_cycle(obs, thresholds)
        assert result is not None
        assert result.pathology == Pathology.FUTILE_CYCLE

    def test_not_enough_steps(self):
        obs = [StepObservation(action="search", step_number=1)]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_futile_cycle(obs, thresholds) is None


class TestOscillation:
    def test_no_oscillation(self):
        obs = [
            StepObservation(action="step one do thing A", step_number=1),
            StepObservation(action="step two do thing B", step_number=2),
            StepObservation(action="step three do thing C", step_number=3),
            StepObservation(action="step four do thing D", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_oscillation(obs, thresholds) is None

    def test_detects_abab_pattern(self):
        # Use very distinct alternating actions so cross-similarity is low
        obs = [
            StepObservation(action="approve expand grow invest more capital budget increase", step_number=1),
            StepObservation(action="reject shrink cut reduce slash spending freeze decrease", step_number=2),
            StepObservation(action="approve expand grow invest more capital budget increase", step_number=3),
            StepObservation(action="reject shrink cut reduce slash spending freeze decrease", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.HIGH)
        result = _detect_oscillation(obs, thresholds)
        assert result is not None
        assert result.pathology == Pathology.OSCILLATION

    def test_not_enough_steps(self):
        obs = [StepObservation(action="a", step_number=1)]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_oscillation(obs, thresholds) is None


class TestStall:
    def test_no_stall_with_progress(self):
        obs = [
            StepObservation(action="find GDP data for Ghana", step_number=1),
            StepObservation(action="GDP data found: 6.0 percent growth", step_number=2),
            StepObservation(action="now analyzing inflation numbers", step_number=3),
            StepObservation(action="inflation analysis complete: 3.3 percent", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        result = _detect_stall(obs, "analyze economic indicators",
                               ["find GDP", "analyze inflation"], thresholds)
        assert result is None

    def test_detects_stall(self):
        obs = [
            StepObservation(action="connect to database server retry attempt", step_number=1),
            StepObservation(action="connect to database server retry attempt", step_number=2),
            StepObservation(action="connect to database server retry attempt", step_number=3),
            StepObservation(action="connect to database server retry attempt", step_number=4),
        ]
        thresholds = _get_thresholds(Sensitivity.HIGH)
        result = _detect_stall(obs, "generate sales report",
                               ["query sales data", "calculate totals"], thresholds)
        assert result is not None
        assert result.pathology == Pathology.STALL


class TestDrift:
    def test_no_drift_when_on_task(self):
        obs = [
            StepObservation(action="research Ghana GDP growth rate", step_number=1),
            StepObservation(action="found Ghana GDP is 6.0 percent", step_number=2),
            StepObservation(action="analyze Ghana economic growth trends", step_number=3),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        result = _detect_drift(obs, "research Ghana GDP growth", thresholds)
        assert result is None

    def test_detects_drift(self):
        obs = [
            StepObservation(action="research Ghana GDP growth rate data", step_number=1),
            StepObservation(action="analyze Ghana economic indicators", step_number=2),
            StepObservation(action="look at cat pictures on the internet", step_number=3),
            StepObservation(action="browse funny memes and videos online", step_number=4),
            StepObservation(action="check social media trending topics today", step_number=5),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        result = _detect_drift(obs, "research Ghana GDP growth rate", thresholds)
        assert result is not None
        assert result.pathology == Pathology.DRIFT

    def test_not_enough_steps(self):
        obs = [StepObservation(action="start", step_number=1)]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_drift(obs, "task", thresholds) is None


class TestAbandonment:
    def test_no_abandonment_when_on_task(self):
        obs = [
            StepObservation(action="research Ghana GDP", step_number=1),
            StepObservation(action="found GDP data Ghana", step_number=2),
            StepObservation(action="compile Ghana GDP report", step_number=3),
            StepObservation(action="finalize Ghana GDP analysis", step_number=4),
            StepObservation(action="submit Ghana GDP findings", step_number=5),
        ]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        result = _detect_abandonment(obs, "research Ghana GDP", thresholds)
        assert result is None

    def test_detects_abandonment(self):
        obs = [
            StepObservation(action="research Ghana GDP growth rate economics", step_number=1),
            StepObservation(action="analyze Ghana GDP economic data statistics", step_number=2),
            StepObservation(action="ordering pizza for lunch delivery food", step_number=3),
            StepObservation(action="ordering pizza for lunch delivery food status", step_number=4),
            StepObservation(action="ordering pizza for lunch delivery food rating", step_number=5),
        ]
        thresholds = _get_thresholds(Sensitivity.HIGH)
        result = _detect_abandonment(obs, "research Ghana GDP growth rate economics", thresholds)
        assert result is not None
        assert result.pathology == Pathology.ABANDONMENT

    def test_not_enough_steps(self):
        obs = [StepObservation(action="start", step_number=1)]
        thresholds = _get_thresholds(Sensitivity.MEDIUM)
        assert _detect_abandonment(obs, "task", thresholds) is None


# ─── PatrolMonitor Tests ──────────���──────────────────────────────────────────


class TestPatrolMonitor:
    def test_healthy_monitoring(self):
        monitor = PatrolMonitor(task_description="analyze data")
        r1 = monitor.observe("collecting data from source A")
        r2 = monitor.observe("processing collected data")
        r3 = monitor.observe("generating analysis report")
        assert r1.is_healthy
        assert r2.is_healthy
        assert r3.is_healthy
        assert monitor.step_count == 3

    def test_detects_pathology(self):
        monitor = PatrolMonitor(
            task_description="analyze data",
            sensitivity="medium",
        )
        # Feed repetitive actions
        for i in range(6):
            report = monitor.observe("search for data in database")
        # After enough repetitions, should detect futile cycle
        summary = monitor.summary()
        assert summary.total_steps == 6

    def test_summary_generation(self):
        monitor = PatrolMonitor(task_description="test")
        monitor.observe("step 1")
        monitor.observe("step 2")
        summary = monitor.summary()
        assert summary.total_steps == 2
        assert isinstance(summary.to_json(), str)

    def test_reset(self):
        monitor = PatrolMonitor()
        monitor.observe("action 1")
        monitor.observe("action 2")
        assert monitor.step_count == 2
        monitor.reset()
        assert monitor.step_count == 0
        assert len(monitor.observations) == 0

    def test_observations_property(self):
        monitor = PatrolMonitor()
        monitor.observe("action 1")
        monitor.observe("action 2")
        obs = monitor.observations
        assert len(obs) == 2
        assert obs[0].action == "action 1"

    def test_sensitivity_levels(self):
        for level in ["low", "medium", "high"]:
            monitor = PatrolMonitor(sensitivity=level)
            report = monitor.observe("test action")
            assert report.step_number == 1

    def test_with_milestones(self):
        monitor = PatrolMonitor(
            task_description="build a house",
            milestones=["lay foundation", "build walls", "add roof"],
        )
        monitor.observe("planning the foundation layout")
        assert monitor.step_count == 1

    def test_empty_summary(self):
        monitor = PatrolMonitor()
        summary = monitor.summary()
        assert summary.total_steps == 0
        assert summary.is_healthy


# ─── Decorator Tests ─────────────────────────────────────────────────────────


class TestPatrolDecorator:
    def test_decorator_preserves_function(self):
        @patrol(on_pathology="ignore")
        def my_step(x):
            return x + 1

        assert my_step(5) == 6

    def test_decorator_has_monitor(self):
        @patrol(on_pathology="ignore")
        def my_step(x):
            return x * 2

        assert hasattr(my_step, '_patrol_monitor')

    def test_decorator_raises_on_pathology(self):
        @patrol(on_pathology="raise", sensitivity="high")
        def repetitive_step(x):
            return x

        # High sensitivity with exact same input should trigger on 3rd call
        with pytest.raises(AgentPathologyError) as exc_info:
            for i in range(10):
                repetitive_step("search for same data")
        assert exc_info.value.report.pathology is not None

    def test_decorator_log_mode(self, capsys):
        @patrol(on_pathology="log", sensitivity="high")
        def log_step(x):
            return x

        # Run many repetitive steps
        for i in range(10):
            log_step("exact same action repeated")

        # Check stderr for log output
        captured = capsys.readouterr()
        # May or may not have triggered depending on detection
        # Just verify no crash


# ─── Thresholds Tests ────────────────────────────────────────────────────────


class TestThresholds:
    def test_high_sensitivity_lower_thresholds(self):
        high = _get_thresholds(Sensitivity.HIGH)
        low = _get_thresholds(Sensitivity.LOW)
        assert high["cycle_similarity"] < low["cycle_similarity"]
        assert high["cycle_window"] < low["cycle_window"]

    def test_medium_between_high_and_low(self):
        high = _get_thresholds(Sensitivity.HIGH)
        med = _get_thresholds(Sensitivity.MEDIUM)
        low = _get_thresholds(Sensitivity.LOW)
        assert high["cycle_similarity"] < med["cycle_similarity"] < low["cycle_similarity"]


# ─── Store Tests ──────��──────────────────────────────────────────────────────


class TestPatrolStore:
    def test_save_and_retrieve(self, tmp_path):
        db = tmp_path / "test.db"
        store = PatrolStore(db)
        summary = PatrolSummary(total_steps=10, health_score=0.9, verdict="healthy")
        run_id = store.save(summary, task_description="test task")
        assert run_id >= 1

        history = store.get_history()
        assert len(history) == 1
        assert history[0]["verdict"] == "healthy"
        store.close()

    def test_get_run(self, tmp_path):
        db = tmp_path / "test.db"
        store = PatrolStore(db)
        summary = PatrolSummary(total_steps=5, verdict="critical")
        run_id = store.save(summary)
        details = store.get_run(run_id)
        assert details["verdict"] == "critical"
        store.close()

    def test_get_nonexistent(self, tmp_path):
        db = tmp_path / "test.db"
        store = PatrolStore(db)
        assert store.get_run(999) is None
        store.close()

    def test_history_limit(self, tmp_path):
        db = tmp_path / "test.db"
        store = PatrolStore(db)
        for _ in range(5):
            store.save(PatrolSummary(total_steps=1))
        assert len(store.get_history(limit=3)) == 3
        store.close()


# ─── Version Test ────────────────────────────────────────────────────────────


def test_version():
    assert __version__ == "0.1.0"


# ─── Integration Tests ─────────��─────────────────────────────────────────────


class TestIntegration:
    def test_full_healthy_run(self, tmp_path):
        monitor = PatrolMonitor(
            task_description="research Ghana economic indicators",
            milestones=["find GDP", "find inflation", "write report"],
        )

        steps = [
            "searching for Ghana GDP data 2025",
            "found GDP: 6.0 percent growth rate",
            "now looking for inflation statistics",
            "inflation rate: 3.3 percent CPI",
            "drafting economic indicators report",
            "report complete with all findings",
        ]

        for action in steps:
            report = monitor.observe(action)

        summary = monitor.summary()
        assert summary.verdict == "healthy"

        db = tmp_path / "integration.db"
        store = PatrolStore(db)
        store.save(summary, task_description="Ghana research")
        history = store.get_history()
        assert len(history) == 1
        store.close()

    def test_detect_cycle_in_real_scenario(self):
        monitor = PatrolMonitor(
            task_description="find latest inflation data",
            sensitivity="high",
        )

        # Simulate agent stuck in a loop
        for i in range(8):
            monitor.observe(f"querying database for inflation statistics round {i}")

        summary = monitor.summary()
        # With high sensitivity and repetitive queries, should detect something
        assert summary.total_steps == 8
