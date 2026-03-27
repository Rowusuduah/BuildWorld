"""LLM judge — evaluates SemanticRules against function outputs."""

from __future__ import annotations

import json
import os
from typing import Any

from llm_contract.models import RuleResult, SemanticRule

# Optional provider imports — None if package not installed
try:
    import anthropic
except ImportError:
    anthropic = None  # type: ignore[assignment]

try:
    import openai
except ImportError:
    openai = None  # type: ignore[assignment]

_JUDGE_SYSTEM_PROMPT = """\
You are a strict behavioral contract evaluator for LLM function outputs.
Your job is to determine whether a given output satisfies a specific behavioral rule.

You will be given:
1. A RULE: a description of what the output must satisfy
2. An OUTPUT: the actual LLM function output to evaluate

Respond with ONLY a JSON object in this exact format:
{
  "passed": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "one-sentence explanation"
}

Rules:
- "passed" must be a boolean
- "confidence" must be a float between 0.0 and 1.0 (how certain you are)
- "reason" must be a single sentence explaining your decision
- Do not include any other text or formatting outside the JSON object
"""

_JUDGE_USER_TEMPLATE = """\
RULE: {rule_description}

OUTPUT:
{output_text}

Does this output satisfy the rule? Respond with JSON only."""


def _serialize_output(output: Any) -> str:
    """Convert any output to a string for judge evaluation."""
    if hasattr(output, "model_dump"):
        return json.dumps(output.model_dump(), indent=2, default=str)
    if hasattr(output, "__dict__"):
        return json.dumps(output.__dict__, indent=2, default=str)
    if isinstance(output, (dict, list)):
        return json.dumps(output, indent=2, default=str)
    return str(output)


def _call_anthropic_judge(
    rule_description: str,
    output_text: str,
    model: str,
) -> dict:
    """Call Anthropic Claude as the LLM judge."""
    if anthropic is None:
        raise ImportError(
            "anthropic package required for Anthropic judge. "
            "Install with: pip install llm-contract[anthropic]"
        )

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    user_message = _JUDGE_USER_TEMPLATE.format(
        rule_description=rule_description,
        output_text=output_text,
    )
    response = client.messages.create(
        model=model,
        max_tokens=256,
        system=_JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    raw = response.content[0].text.strip()
    return json.loads(raw)


def _call_openai_judge(
    rule_description: str,
    output_text: str,
    model: str,
) -> dict:
    """Call OpenAI as the LLM judge."""
    if openai is None:
        raise ImportError(
            "openai package required for OpenAI judge. "
            "Install with: pip install llm-contract[openai]"
        )

    client = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    user_message = _JUDGE_USER_TEMPLATE.format(
        rule_description=rule_description,
        output_text=output_text,
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=256,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content.strip()
    return json.loads(raw)


def evaluate_rule(
    rule: SemanticRule,
    output: Any,
    provider: str,
    model: str,
) -> RuleResult:
    """Evaluate a single SemanticRule against an LLM function output.

    Args:
        rule: The behavioral rule to evaluate.
        output: The output from the LLM function.
        provider: Judge provider — ``"anthropic"`` or ``"openai"``.
        model: Judge model identifier.

    Returns:
        A :class:`RuleResult` with the judge's verdict.
    """
    if not rule.enabled:
        return RuleResult(
            rule_name=rule.name,
            passed=True,
            confidence=1.0,
            reason="Rule disabled — skipped.",
            weight=rule.weight,
        )

    output_text = _serialize_output(output)

    if provider == "anthropic":
        verdict = _call_anthropic_judge(rule.description, output_text, model)
    elif provider == "openai":
        verdict = _call_openai_judge(rule.description, output_text, model)
    else:
        raise ValueError(
            f"Unsupported judge provider: {provider!r}. Use 'anthropic' or 'openai'."
        )

    passed = bool(verdict.get("passed", False))
    confidence = float(verdict.get("confidence", 0.0))
    reason = str(verdict.get("reason", "No reason provided."))

    # Apply per-rule threshold
    if confidence < rule.threshold:
        passed = False
        reason = f"{reason} [confidence {confidence:.2f} < threshold {rule.threshold:.2f}]"

    return RuleResult(
        rule_name=rule.name,
        passed=passed,
        confidence=confidence,
        reason=reason,
        weight=rule.weight,
    )


def compute_overall_score(rule_results: list[RuleResult]) -> float:
    """Compute weighted average score across all rule results.

    Critical rules (weight=1.0) that fail will pull the score to 0.
    """
    if not rule_results:
        return 1.0

    total_weight = sum(r.weight for r in rule_results)
    if total_weight == 0:
        return 1.0

    weighted_sum = sum(
        r.confidence * r.weight if r.passed else 0.0
        for r in rule_results
    )
    return weighted_sum / total_weight
