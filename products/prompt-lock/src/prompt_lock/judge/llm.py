"""LLM-as-a-judge scoring via LiteLLM (any provider)."""

from __future__ import annotations

import json
import re

JUDGE_PROMPT = """\
You are an impartial evaluator assessing the quality of an AI assistant's response.

EVALUATION CRITERIA:
{criteria}

USER INPUT:
{input}

RESPONSE TO EVALUATE:
{output}

Score the response from 0.0 to 1.0:
  0.0 = completely fails the criteria
  0.5 = partially meets the criteria
  1.0 = fully meets the criteria

Respond with ONLY a JSON object in this exact format (no other text):
{{"score": <number>, "reasoning": "<one sentence>"}}"""


def llm_judge_score(
    input_text: str,
    output_text: str,
    criteria: str,
    model: str = "gpt-4o-mini",
) -> tuple[float, str]:
    """Score an output using an LLM judge. Returns (score, reasoning).

    Uses LiteLLM so any provider (OpenAI, Anthropic, Mistral, local) works.
    """
    import litellm

    prompt = JUDGE_PROMPT.format(
        criteria=criteria,
        input=input_text,
        output=output_text,
    )
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=256,
    )
    content = response.choices[0].message.content.strip()

    try:
        data = json.loads(content)
        score = float(data["score"])
        reasoning = str(data.get("reasoning", ""))
        return max(0.0, min(1.0, score)), reasoning
    except (json.JSONDecodeError, KeyError, ValueError):
        # Fallback: extract first float-looking number from the response
        match = re.search(r"\b(0\.\d+|1\.0+|0|1)\b", content)
        if match:
            return max(0.0, min(1.0, float(match.group()))), content
        return 0.5, content
