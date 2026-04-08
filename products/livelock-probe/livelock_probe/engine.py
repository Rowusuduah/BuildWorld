"""
livelock_probe.engine
---------------------
Core livelock detection engine.

Algorithm (PAT-075 — John 5:5-9, The 38-Year Stuck State Pattern):
  1. For each agent step output, compute cosine similarity to the goal description.
  2. Build a progress_vector: [sim(step_0, goal), sim(step_1, goal), ...]
  3. Build progress_deltas: delta[0] = progress_vector[0];
                            delta[i] = progress_vector[i] - progress_vector[i-1]
  4. A step is "stuck" if |delta[i]| < epsilon.
  5. livelock_score = fraction of stuck steps in [0.0, 1.0].
  6. livelock_detected = max_consecutive_stuck >= k.

Default similarity: smoothed TF-IDF cosine (zero external dependencies).
Injectable similarity_fn: accepts any callable(str, str) -> float in [0, 1].
Neural option: sentence-transformers all-MiniLM-L6-v2 (optional install).
"""
from __future__ import annotations

import math
import re
import uuid
from collections import Counter
from datetime import datetime, timezone
from typing import Callable, List, Optional, Tuple

from .models import (
    CriticalityLevel,
    LivelockReport,
    LivelockVerdict,
    ProgressConfig,
    StepRecord,
    get_threshold,
    make_recommendation,
    score_to_verdict,
)


# ── TF-IDF Cosine Similarity (zero dependencies) ─────────────────────────────

def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokenizer."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tfidf_cosine(text_a: str, text_b: str) -> float:
    """
    Smoothed TF-IDF cosine similarity between two texts.
    Zero external dependencies — uses stdlib only.
    Returns value in [0.0, 1.0].

    Uses add-1 smoothed IDF to prevent near-zero weights for terms
    shared across both documents in a small 2-document corpus.
    """
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    tf_a = Counter(tokens_a)
    tf_b = Counter(tokens_b)
    vocab = set(tf_a) | set(tf_b)

    idf: dict[str, float] = {}
    for term in vocab:
        df = sum(1 for tf in (tf_a, tf_b) if term in tf)
        idf[term] = math.log(1.0 + 2.0 / df)

    total_a = sum(tf_a.values()) or 1
    total_b = sum(tf_b.values()) or 1

    vec_a = {t: (tf_a[t] / total_a) * idf[t] for t in tf_a}
    vec_b = {t: (tf_b[t] / total_b) * idf[t] for t in tf_b}

    dot = sum(vec_a.get(t, 0.0) * vec_b.get(t, 0.0) for t in vocab)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values())) or 1e-9
    norm_b = math.sqrt(sum(v * v for v in vec_b.values())) or 1e-9

    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


def _neural_cosine(text_a: str, text_b: str) -> float:
    """
    Neural cosine similarity via sentence-transformers all-MiniLM-L6-v2.
    Falls back to TF-IDF if sentence-transformers is not installed.
    """
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        import numpy as np  # type: ignore

        model = SentenceTransformer("all-MiniLM-L6-v2")
        emb_a, emb_b = model.encode([text_a, text_b], normalize_embeddings=True)
        cos = float(np.dot(emb_a, emb_b))
        return max(0.0, min(1.0, cos))
    except ImportError:
        return _tfidf_cosine(text_a, text_b)


# ── Progress Vector Computation ───────────────────────────────────────────────

def _compute_progress_vector(
    steps: List[str],
    goal: str,
    similarity_fn: Callable[[str, str], float],
) -> List[float]:
    """
    Compute progress_to_goal for each step output.

    Returns:
        List of similarity scores in [0.0, 1.0], one per step.
        Empty list if steps is empty.
    """
    return [similarity_fn(step, goal) for step in steps]


def _compute_progress_deltas(progress_vector: List[float]) -> List[float]:
    """
    Compute per-step progress deltas.

    delta[0] = progress_vector[0]  (initial progress from baseline 0)
    delta[i] = progress_vector[i] - progress_vector[i-1]  for i > 0

    Returns:
        List of floats, same length as progress_vector.
    """
    if not progress_vector:
        return []
    deltas: List[float] = [progress_vector[0]]
    for i in range(1, len(progress_vector)):
        deltas.append(progress_vector[i] - progress_vector[i - 1])
    return deltas


def _find_max_consecutive_stuck(stuck_mask: List[bool]) -> int:
    """Return the length of the longest run of True values in stuck_mask."""
    max_run = 0
    current_run = 0
    for val in stuck_mask:
        if val:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0
    return max_run


def _find_stuck_window(
    stuck_mask: List[bool],
) -> Tuple[Optional[int], Optional[int]]:
    """
    Find the start and end indices of the longest consecutive stuck window.

    Returns:
        (start, end) tuple where both are inclusive step indices,
        or (None, None) if no stuck window exists.
    """
    best_start: Optional[int] = None
    best_end: Optional[int] = None
    best_len = 0

    current_start: Optional[int] = None
    current_len = 0

    for i, val in enumerate(stuck_mask):
        if val:
            if current_start is None:
                current_start = i
            current_len += 1
            if current_len > best_len:
                best_len = current_len
                best_start = current_start
                best_end = i
        else:
            current_start = None
            current_len = 0

    return best_start, best_end


# ── LivelockEngine ────────────────────────────────────────────────────────────

class LivelockEngine:
    """
    Computes LivelockScore and produces a LivelockReport from agent step outputs.

    Usage (zero dependencies, TF-IDF default):
        engine = LivelockEngine()
        report = engine.compute(
            steps=["I searched the docs.", "I searched the docs again.",
                   "I searched the docs once more.", "Docs found nothing.",
                   "Retrying search.", "Retrying again."],
            config=ProgressConfig(goal="resolve the database connection error", k=3),
        )
        print(report.summary())

    Usage (injectable for testing):
        engine = LivelockEngine(similarity_fn=lambda a, b: 0.95 if "resolve" in a else 0.01)

    Usage (neural):
        engine = LivelockEngine(use_neural=True)
    """

    def __init__(
        self,
        *,
        use_neural: bool = False,
        similarity_fn: Optional[Callable[[str, str], float]] = None,
    ) -> None:
        """
        Args:
            use_neural: Use sentence-transformers for embeddings.
            similarity_fn: Injectable similarity function. Overrides neural/TF-IDF.
        """
        self._similarity_fn = similarity_fn
        self.use_neural = use_neural

    def _get_similarity_fn(
        self,
        config: ProgressConfig,
    ) -> Callable[[str, str], float]:
        """Resolve the similarity function from engine settings + config."""
        if config.similarity_fn is not None:
            return config.similarity_fn
        if self._similarity_fn is not None:
            return self._similarity_fn
        if config.use_neural or self.use_neural:
            return _neural_cosine
        return _tfidf_cosine

    def compute(
        self,
        steps: List[str],
        config: ProgressConfig,
    ) -> LivelockReport:
        """
        Compute a LivelockReport from a list of agent step outputs.

        Args:
            steps: List of agent step outputs (strings), in order.
                   Each string represents what the agent produced at that step.
            config: ProgressConfig specifying the goal, k, epsilon, and criticality.

        Returns:
            LivelockReport with full livelock analysis.

        Raises:
            ValueError: If steps is empty.
        """
        if not steps:
            raise ValueError(
                "livelock-probe requires at least 1 step output to evaluate. "
                "Got 0 steps."
            )

        sim_fn = self._get_similarity_fn(config)

        # 1. Compute progress vector (similarity to goal per step)
        progress_vector = _compute_progress_vector(steps, config.goal, sim_fn)

        # 2. Compute progress deltas
        progress_deltas = _compute_progress_deltas(progress_vector)

        # 3. Build stuck mask: True if |delta| < epsilon
        stuck_mask = [abs(d) < config.epsilon for d in progress_deltas]

        # 4. Build step records
        step_records: List[StepRecord] = [
            StepRecord(
                step_id=i,
                output=steps[i],
                progress_to_goal=progress_vector[i],
                progress_delta=progress_deltas[i],
                is_stuck=stuck_mask[i],
            )
            for i in range(len(steps))
        ]

        # 5. Compute livelock score
        livelock_score = sum(stuck_mask) / len(stuck_mask)

        # 6. Detect livelock
        max_consecutive_stuck = _find_max_consecutive_stuck(stuck_mask)
        livelock_detected = max_consecutive_stuck >= config.k

        # 7. Find stuck window
        stuck_window_start, stuck_window_end = _find_stuck_window(stuck_mask)

        # 8. Compute mean progress
        mean_progress = sum(progress_deltas) / len(progress_deltas)

        # 9. Compute verdict and gate
        threshold = get_threshold(config.criticality)
        verdict = score_to_verdict(livelock_score, config.criticality, config.borderline_band)
        gate_passed = livelock_score <= threshold

        # 10. Generate recommendation
        recommendation = make_recommendation(
            livelock_score=livelock_score,
            livelock_detected=livelock_detected,
            max_consecutive_stuck=max_consecutive_stuck,
            k=config.k,
            stuck_window_start=stuck_window_start,
        )

        return LivelockReport(
            report_id=str(uuid.uuid4()),
            goal=config.goal,
            livelock_score=livelock_score,
            livelock_detected=livelock_detected,
            stuck_window_start=stuck_window_start,
            stuck_window_end=stuck_window_end,
            total_steps=len(steps),
            progress_vector=progress_vector,
            progress_deltas=progress_deltas,
            mean_progress=mean_progress,
            max_consecutive_stuck=max_consecutive_stuck,
            gate_passed=gate_passed,
            verdict=verdict,
            steps=step_records,
            recommendation=recommendation,
            criticality=config.criticality,
            threshold=threshold,
            k=config.k,
            epsilon=config.epsilon,
            agent_label=config.agent_label,
            tested_at=datetime.now(timezone.utc),
        )
