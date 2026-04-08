"""
pressure-gauge engine
---------------------
TF-IDF cosine similarity, context padding, and pressure sweep computation.
Zero external dependencies for core functionality.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Callable, Dict, List, Optional

from .models import (
    DriftPoint,
    PressureConfig,
    PressureReport,
    get_threshold,
    score_to_verdict,
)

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def approx_token_count(text: str, chars_per_token: float = 4.0) -> int:
    """Approximate token count using character-based heuristic."""
    return max(1, int(len(text) / chars_per_token))


# ---------------------------------------------------------------------------
# Context padding
# ---------------------------------------------------------------------------

_LOREM = (
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
    "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
    "Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris "
    "nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in "
    "reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. "
    "Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia "
    "deserunt mollit anim id est laborum. "
)

_HISTORY_ENTRY = (
    "User: I need help with the task at hand.\n"
    "Assistant: I understand your request and I am working on it carefully. "
    "Let me analyze the requirements and provide a thorough and complete response "
    "that addresses all aspects of the problem you have described.\n"
)


def generate_padding(
    target_tokens: int,
    strategy: str,
    chars_per_token: float = 4.0,
    custom_text: Optional[str] = None,
) -> str:
    """Generate padding text approximating target_tokens tokens."""
    target_chars = int(target_tokens * chars_per_token)
    if target_chars <= 0:
        return ""

    if strategy == "lorem_ipsum":
        base = _LOREM
    elif strategy == "repeat_text":
        base = custom_text if custom_text else _LOREM
    elif strategy == "inject_history":
        base = _HISTORY_ENTRY
    else:
        base = _LOREM

    repetitions = max(1, math.ceil(target_chars / len(base)))
    return (base * repetitions)[:target_chars]


def build_padded_context(
    base_context: str,
    target_tokens: int,
    strategy: str,
    chars_per_token: float = 4.0,
    custom_text: Optional[str] = None,
) -> str:
    """
    Pad base_context with noise to reach approximately target_tokens.
    If base_context already meets or exceeds target_tokens, return as-is.
    Padding is injected before base_context (simulating filled history).
    """
    current_tokens = approx_token_count(base_context, chars_per_token)
    if current_tokens >= target_tokens:
        return base_context

    needed = target_tokens - current_tokens
    padding = generate_padding(needed, strategy, chars_per_token, custom_text)
    return padding + "\n\n" + base_context


# ---------------------------------------------------------------------------
# TF-IDF cosine similarity (zero external dependencies)
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> List[str]:
    """Lowercase whitespace + alphanumeric tokenizer."""
    return re.findall(r"[a-z0-9']+", text.lower())


def _compute_tfidf(docs: List[str]) -> List[Dict[str, float]]:
    """Return TF-IDF vector per document."""
    n = len(docs)
    if n == 0:
        return []

    tokenized = [_tokenize(doc) for doc in docs]

    # Document frequency
    df: Counter = Counter()
    for tokens in tokenized:
        df.update(set(tokens))

    vectors: List[Dict[str, float]] = []
    for tokens in tokenized:
        if not tokens:
            vectors.append({})
            continue
        tf = Counter(tokens)
        total = len(tokens)
        vec: Dict[str, float] = {}
        for term, count in tf.items():
            tf_val = count / total
            idf_val = math.log((n + 1) / (df[term] + 1)) + 1.0  # smoothed IDF
            vec[term] = tf_val * idf_val
        vectors.append(vec)

    return vectors


def cosine_similarity(
    vec_a: Dict[str, float], vec_b: Dict[str, float]
) -> float:
    """Cosine similarity between two sparse TF-IDF vectors."""
    if not vec_a or not vec_b:
        return 0.0

    common = set(vec_a.keys()) & set(vec_b.keys())
    dot = sum(vec_a[k] * vec_b[k] for k in common)
    norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
    norm_b = math.sqrt(sum(v * v for v in vec_b.values()))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return min(1.0, dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Optional neural embeddings (sentence-transformers)
# ---------------------------------------------------------------------------

def _try_neural_embed(texts: List[str]) -> Optional[List[List[float]]]:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
        model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = model.encode(texts, convert_to_numpy=True)
        return [list(emb) for emb in embeddings]
    except ImportError:
        return None


def _neural_cosine(vec_a: List[float], vec_b: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return min(1.0, dot / (norm_a * norm_b))


# ---------------------------------------------------------------------------
# Core pressure computation
# ---------------------------------------------------------------------------

def compute_similarities(
    outputs: List[str],
    use_neural: bool = False,
) -> List[float]:
    """
    Compute cosine similarity of each output to the first (baseline) output.
    Returns a list of floats, same length as outputs; first element = ~1.0.
    """
    if not outputs:
        return []

    if use_neural:
        embeddings = _try_neural_embed(outputs)
        if embeddings is not None:
            baseline = embeddings[0]
            return [_neural_cosine(baseline, emb) for emb in embeddings]

    # Fallback: TF-IDF cosine
    vectors = _compute_tfidf(outputs)
    if not vectors:
        return [1.0] * len(outputs)

    baseline = vectors[0]
    return [cosine_similarity(baseline, vec) for vec in vectors]


def run_sweep(
    config: PressureConfig,
    agent_fn: Callable[[str], str],
    base_context: str = "",
    use_neural: bool = False,
) -> PressureReport:
    """
    Execute the full context pressure sweep.

    agent_fn(context: str) -> str
        Receives padded context string. Returns agent output string.
    base_context
        The task/query appended after the padding at every fill level.
    """
    outputs_per_level: List[List[str]] = []

    for fill_level in config.fill_levels:
        target_tokens = config.tokens_for_level(fill_level)
        padded = build_padded_context(
            base_context,
            target_tokens,
            config.padding_strategy,
            config.chars_per_token,
            config.padding_text,
        )

        level_outputs: List[str] = []
        for _ in range(config.runs_per_level):
            output = agent_fn(padded)
            level_outputs.append(output if output else "")
        outputs_per_level.append(level_outputs)

    # Representative output per level: join multiple runs for embedding
    representative: List[str] = [
        " ".join(outs) for outs in outputs_per_level
    ]

    similarities = compute_similarities(representative, use_neural=use_neural)

    drift_curve: List[DriftPoint] = []
    for fill_level, sim, level_outs in zip(
        config.fill_levels, similarities, outputs_per_level
    ):
        token_count = config.tokens_for_level(fill_level)
        verdict = score_to_verdict(sim, config.criticality)
        drift_curve.append(
            DriftPoint(
                fill_level=fill_level,
                token_count=token_count,
                similarity_to_baseline=sim,
                verdict=verdict,
                outputs=level_outs,
            )
        )

    return _build_report(config, drift_curve)


def _build_report(
    config: PressureConfig, drift_curve: List[DriftPoint]
) -> PressureReport:
    """Assemble PressureReport from the computed drift curve."""
    threshold = get_threshold(config.criticality)

    # ContextPressureScore = mean similarity of non-baseline levels to baseline
    non_baseline = drift_curve[1:] if len(drift_curve) > 1 else drift_curve
    if non_baseline:
        score = sum(dp.similarity_to_baseline for dp in non_baseline) / len(
            non_baseline
        )
    else:
        score = (
            drift_curve[0].similarity_to_baseline if drift_curve else 1.0
        )

    # pressure_onset_token: first fill level (excluding baseline) where sim < threshold
    onset_token: Optional[int] = None
    for dp in drift_curve[1:]:
        if dp.similarity_to_baseline < threshold:
            onset_token = dp.token_count
            break

    gate_passed = score >= threshold
    verdict = score_to_verdict(score, config.criticality)

    if gate_passed:
        recommendation = (
            f"Agent behavior is stable across context fill levels "
            f"(ContextPressureScore={score:.3f} >= {threshold}). "
            "No context anxiety detected."
        )
    else:
        onset_str = f" at ~{onset_token:,} tokens" if onset_token else ""
        recommendation = (
            f"Context pressure detected{onset_str}: agent behavior drifts as "
            f"context window fills. ContextPressureScore={score:.3f} < "
            f"{threshold} ({config.criticality.value}). "
            "Consider: context compression, explicit task reminders injected "
            "near the end of context, or reducing context window utilization."
        )

    return PressureReport(
        config=config,
        drift_curve=drift_curve,
        context_pressure_score=score,
        pressure_onset_token=onset_token,
        verdict=verdict,
        gate_passed=gate_passed,
        recommendation=recommendation,
    )
