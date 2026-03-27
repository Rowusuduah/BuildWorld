"""Eval runners: exact_match, regex, semantic_similarity, llm_judge, custom."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .config import EvalConfig


@dataclass
class EvalResult:
    eval_type: str
    score: float
    passed: bool
    threshold: float
    details: str
    input_text: str
    output_text: str


# Singleton sentence-transformer model (lazy-loaded, cached)
_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        from sentence_transformers import SentenceTransformer

        _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _sentence_model


def run_exact_match(
    input_text: str,
    output_text: str,
    expected: str,
    threshold: float = 1.0,
) -> EvalResult:
    score = 1.0 if output_text.strip() == expected.strip() else 0.0
    return EvalResult(
        eval_type="exact_match",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=f"expected={expected[:60]!r}",
        input_text=input_text,
        output_text=output_text,
    )


def run_regex(
    input_text: str,
    output_text: str,
    pattern: str,
    threshold: float = 1.0,
) -> EvalResult:
    match = re.search(pattern, output_text, re.DOTALL)
    score = 1.0 if match else 0.0
    return EvalResult(
        eval_type="regex",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=f"pattern={pattern!r}, matched={bool(match)}",
        input_text=input_text,
        output_text=output_text,
    )


def run_semantic_similarity(
    input_text: str,
    output_text: str,
    expected: str,
    threshold: float = 0.80,
) -> EvalResult:
    from sentence_transformers.util import cos_sim

    model = _get_sentence_model()
    emb_out = model.encode(output_text, convert_to_tensor=True)
    emb_exp = model.encode(expected, convert_to_tensor=True)
    score = float(cos_sim(emb_out, emb_exp))
    score = max(0.0, min(1.0, score))

    return EvalResult(
        eval_type="semantic_similarity",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=f"cosine_similarity={score:.4f}",
        input_text=input_text,
        output_text=output_text,
    )


def run_llm_judge(
    input_text: str,
    output_text: str,
    criteria: str,
    model: str = "gpt-4o-mini",
    threshold: float = 0.70,
) -> EvalResult:
    from .judge.llm import llm_judge_score

    score, reasoning = llm_judge_score(input_text, output_text, criteria, model)
    return EvalResult(
        eval_type="llm_judge",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=reasoning,
        input_text=input_text,
        output_text=output_text,
    )


def run_custom(
    input_text: str,
    output_text: str,
    custom_fn: str,
    threshold: float = 0.70,
) -> EvalResult:
    """Run a user-defined eval function. custom_fn is 'module.function_name'.

    The function must accept (input: str, output: str) and return a float 0.0–1.0.
    """
    import importlib

    module_path, fn_name = custom_fn.rsplit(".", 1)
    module = importlib.import_module(module_path)
    fn = getattr(module, fn_name)
    score = float(fn(input_text, output_text))
    score = max(0.0, min(1.0, score))

    return EvalResult(
        eval_type="custom",
        score=score,
        passed=score >= threshold,
        threshold=threshold,
        details=f"fn={custom_fn}",
        input_text=input_text,
        output_text=output_text,
    )


def run_eval(
    eval_config: EvalConfig,
    input_text: str,
    output_text: str,
    expected: Optional[str] = None,
    default_model: str = "gpt-4o-mini",
) -> EvalResult:
    """Dispatch to the correct eval runner based on config type."""
    t = eval_config.type

    if t == "exact_match":
        if not expected:
            raise ValueError("exact_match requires 'expected_output' in the test case")
        return run_exact_match(input_text, output_text, expected, eval_config.threshold)

    elif t == "regex":
        if not eval_config.pattern:
            raise ValueError("regex eval requires 'pattern' in the eval config")
        return run_regex(input_text, output_text, eval_config.pattern, eval_config.threshold)

    elif t == "semantic_similarity":
        if not expected:
            raise ValueError("semantic_similarity requires 'expected_output' in the test case")
        return run_semantic_similarity(
            input_text, output_text, expected, eval_config.threshold
        )

    elif t == "llm_judge":
        if not eval_config.criteria:
            raise ValueError("llm_judge eval requires 'criteria' in the eval config")
        model = eval_config.model or default_model
        return run_llm_judge(
            input_text, output_text, eval_config.criteria, model, eval_config.threshold
        )

    elif t == "custom":
        if not eval_config.custom_fn:
            raise ValueError("custom eval requires 'custom_fn' in the eval config")
        return run_custom(
            input_text, output_text, eval_config.custom_fn, eval_config.threshold
        )

    else:
        raise ValueError(f"Unknown eval type: {t!r}")


def load_test_cases(path: str | Path) -> list[dict]:
    """Load test cases from a JSONL file.

    Each line: {"input": "...", "output": "...", "expected_output": "..."}
    'output' is the actual model output to evaluate.
    'expected_output' is used by exact_match and semantic_similarity evals.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Test cases file not found: {path}")

    cases = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")
            if "input" not in record:
                raise ValueError(f"Line {i} of {path} missing 'input' key")
            cases.append(record)

    return cases
