"""context_trace.runners
~~~~~~~~~~~~~~~~~~~~~
Pre-built runner adapters for common LLM providers.

A runner is any Callable[[str], str] — takes a prompt, returns a response.
These factory functions build runners backed by real provider SDKs.

Usage::

    from context_trace.runners import anthropic_runner, openai_runner

    runner = anthropic_runner(model="claude-haiku-4-5-20251001")
    tracer = ContextTracer(runner=runner, k=3)
"""

from __future__ import annotations

from typing import Callable, Optional

RunnerFn = Callable[[str], str]


def anthropic_runner(
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 512,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
) -> RunnerFn:
    """
    Build a runner function backed by the Anthropic Messages API.

    Args:
        model: Anthropic model ID.
        max_tokens: Maximum tokens in response.
        system: Optional system prompt.
        api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var).
        temperature: Sampling temperature (higher = more variance in attribution).

    Returns:
        Callable[[str], str] for use as ContextTracer.runner.
    """
    def _run(prompt: str) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        kwargs: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        if system:
            kwargs["system"] = system
        response = client.messages.create(**kwargs)
        return response.content[0].text

    return _run


def openai_runner(
    model: str = "gpt-4o-mini",
    max_tokens: int = 512,
    system: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.7,
) -> RunnerFn:
    """
    Build a runner function backed by the OpenAI Chat Completions API.

    Args:
        model: OpenAI model ID.
        max_tokens: Maximum tokens in response.
        system: Optional system message.
        api_key: OpenAI API key (defaults to OPENAI_API_KEY env var).
        temperature: Sampling temperature.

    Returns:
        Callable[[str], str] for use as ContextTracer.runner.
    """
    def _run(prompt: str) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature,
        )
        return response.choices[0].message.content

    return _run
