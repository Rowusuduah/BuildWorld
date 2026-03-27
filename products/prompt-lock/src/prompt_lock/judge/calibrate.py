"""Judge calibration: verify LLM judge agrees with human labels before trusting it as a CI gate.

This is the key differentiator of prompt-lock. No other tool ships this as a pip-installable
CI primitive that can block a deployment when the judge is miscalibrated.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .llm import llm_judge_score


@dataclass
class CalibrationResult:
    passed: bool
    agreement_rate: float
    spearman_correlation: float
    bias: float  # mean(judge_scores) - mean(human_scores); positive = judge inflates
    n_examples: int
    model: str
    criteria: str
    details: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"Calibration {status}: "
            f"agreement={self.agreement_rate:.1%}, "
            f"spearman={self.spearman_correlation:.3f}, "
            f"bias={self.bias:+.3f}, "
            f"n={self.n_examples}, "
            f"model={self.model}"
        )


def load_human_labels(path: str | Path) -> list[dict]:
    """Load human-labeled examples from a JSONL file.

    Each line must be JSON with keys: input, output, human_score (float 0.0–1.0).
    Example:
        {"input": "What is 2+2?", "output": "4", "human_score": 1.0}
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Human labels file not found: {path}\n"
            "Create a JSONL file with lines like:\n"
            '  {"input": "...", "output": "...", "human_score": 0.9}'
        )

    examples = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {i} of {path}: {e}")

            required = {"input", "output", "human_score"}
            missing = required - record.keys()
            if missing:
                raise ValueError(f"Line {i} of {path} missing required keys: {missing}")

            human_score = float(record["human_score"])
            if not (0.0 <= human_score <= 1.0):
                raise ValueError(
                    f"Line {i}: human_score={human_score} must be between 0.0 and 1.0"
                )
            examples.append(record)

    return examples


def calibrate_judge(
    human_labels_file: str | Path,
    model: str,
    criteria: str,
    min_agreement: float = 0.80,
    min_spearman: float = 0.70,
    agreement_tolerance: float = 0.15,
) -> CalibrationResult:
    """Run calibration: score each human-labeled example with the LLM judge and measure alignment.

    Passes if:
        agreement_rate >= min_agreement  (fraction of examples within ±tolerance)
        spearman_correlation >= min_spearman

    Args:
        human_labels_file: JSONL file with {input, output, human_score} records.
        model: LiteLLM model string (e.g. "gpt-4o-mini", "claude-haiku-4-5-20251001").
        criteria: The evaluation criteria string the judge uses.
        min_agreement: Minimum fraction of examples where judge ≈ human (within tolerance).
        min_spearman: Minimum Spearman rank correlation between judge and human scores.
        agreement_tolerance: Two scores are "in agreement" if |judge - human| <= tolerance.

    Returns:
        CalibrationResult with passed=True if all thresholds are met.
    """
    from scipy.stats import spearmanr

    examples = load_human_labels(human_labels_file)

    if len(examples) < 5:
        raise ValueError(
            f"Need at least 5 human-labeled examples for calibration, got {len(examples)}.\n"
            f"Add more examples to {human_labels_file}"
        )

    judge_scores: list[float] = []
    human_scores: list[float] = []
    details: list[dict] = []

    for ex in examples:
        judge_score, reasoning = llm_judge_score(
            input_text=ex["input"],
            output_text=ex["output"],
            criteria=criteria,
            model=model,
        )
        judge_scores.append(judge_score)
        human_scores.append(float(ex["human_score"]))
        in_agreement = abs(judge_score - float(ex["human_score"])) <= agreement_tolerance
        details.append(
            {
                "input": ex["input"][:120],
                "human_score": ex["human_score"],
                "judge_score": judge_score,
                "reasoning": reasoning,
                "agreement": in_agreement,
            }
        )

    agreement_rate = sum(1 for d in details if d["agreement"]) / len(examples)

    corr_result = spearmanr(judge_scores, human_scores)
    spearman = float(corr_result.statistic)
    if spearman != spearman:  # NaN guard (e.g. all same scores)
        spearman = 0.0

    bias = statistics.mean(judge_scores) - statistics.mean(human_scores)

    passed = agreement_rate >= min_agreement and spearman >= min_spearman

    return CalibrationResult(
        passed=passed,
        agreement_rate=agreement_rate,
        spearman_correlation=spearman,
        bias=bias,
        n_examples=len(examples),
        model=model,
        criteria=criteria,
        details=details,
    )
