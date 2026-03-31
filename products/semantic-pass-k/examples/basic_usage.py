"""
basic_usage.py
--------------
Basic usage examples for semantic-pass-k.
All examples run without any LLM or external dependencies.
"""
from semantic_pass_k import ConsistencyEngine, ConsistencyRunner
from semantic_pass_k.models import CRITICALITY_THRESHOLDS


# ── Example 1: Direct engine evaluation ───────────────────────────────────────

def example_direct_evaluation():
    """Evaluate consistency from pre-collected outputs."""
    print("=" * 50)
    print("Example 1: Direct evaluation")
    print("=" * 50)

    engine = ConsistencyEngine()

    # Lexically similar outputs — consistent agent (TF-IDF handles these well)
    outputs_consistent = [
        "Accra is the capital of Ghana.",
        "The capital of Ghana is Accra.",
        "Ghana capital is Accra city.",
        "Accra Ghana capital city.",
        "Ghana's capital city is Accra.",
    ]

    result = engine.evaluate(
        prompt="What is the capital of Ghana?",
        outputs=outputs_consistent,
        criticality="HIGH",
    )
    print(f"Consistent agent: {result.summary()}")

    # Semantically divergent outputs (inconsistent agent)
    outputs_inconsistent = [
        "Ghana's capital is Accra.",
        "The GDP growth rate is 6% according to IMF projections.",
        "The policy requires environmental impact assessments.",
        "Market hours are 9am to 5pm on weekdays.",
        "The rainfall season starts in April.",
    ]

    result2 = engine.evaluate(
        prompt="What is the capital of Ghana?",
        outputs=outputs_inconsistent,
        criticality="HIGH",
    )
    print(f"Inconsistent agent: {result2.summary()}")


# ── Example 2: Runner with stub agent ─────────────────────────────────────────

def example_with_runner():
    """Use ConsistencyRunner with a stubbed agent function."""
    print("\n" + "=" * 50)
    print("Example 2: ConsistencyRunner")
    print("=" * 50)

    # Stub agent that always returns a consistent answer
    def my_stub_agent(prompt: str) -> str:
        return f"The answer to '{prompt}' is consistent every time."

    runner = ConsistencyRunner(
        agent_fn=my_stub_agent,
        k=5,
        criticality="HIGH",
    )

    result = runner.run("What is your answer?")
    print(f"Runner result: {result.summary()}")


# ── Example 3: Batch evaluation ───────────────────────────────────────────────

def example_batch_evaluation():
    """Evaluate consistency across multiple prompts."""
    print("\n" + "=" * 50)
    print("Example 3: Batch evaluation")
    print("=" * 50)

    engine = ConsistencyEngine()

    prompts = [
        "What is the capital of Ghana?",
        "What is 2 + 2?",
    ]
    outputs_per_prompt = [
        ["Accra is the capital.", "Ghana's capital is Accra.", "The capital city is Accra."],
        ["2 + 2 = 4.", "The answer is 4.", "Four.", "2+2 equals 4.", "It's 4."],
    ]

    results = engine.evaluate_batch(prompts, outputs_per_prompt, criticality="MEDIUM")
    for r in results:
        print(f"  Prompt: '{r.prompt[:40]}...' -> {r.summary()}")


# ── Example 4: Criticality tiers ─────────────────────────────────────────────

def example_criticality_tiers():
    """Show how the same score can pass/fail different criticality tiers."""
    print("\n" + "=" * 50)
    print("Example 4: Criticality tiers")
    print("=" * 50)

    # Inject a fixed similarity of 0.92 (passes HIGH but fails CRITICAL)
    engine = ConsistencyEngine(similarity_fn=lambda a, b: 0.92)
    outputs = ["output a", "output b"]

    for tier in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        result = engine.evaluate("prompt", outputs, tier)  # type: ignore
        threshold = CRITICALITY_THRESHOLDS[tier]
        status = "PASS" if result.passed else "FAIL"
        print(
            f"  {tier:8s}  threshold={threshold:.2f}  "
            f"score={result.consistency_score:.3f}  "
            f"verdict={result.verdict:12s}  [{status}]"
        )


if __name__ == "__main__":
    example_direct_evaluation()
    example_with_runner()
    example_batch_evaluation()
    example_criticality_tiers()
